import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
import equinox as eqx
from collections.abc import Callable

import skfem as fem
from skfem.helpers import dot, grad
import warnings

from ormatex_py.progression import element_line_pp_nodal as el_nodal
from ormatex_py.ode_sys import JaxMatrixLinop

jax.config.update("jax_enable_x64", True)

from ormatex_py.ode_sys import OdeSys
from ormatex_py.ode_epirk import EpirkIntegrator


@fem.BilinearForm
def adv_diff(u, v, w):
    """
    Combo Adv Diff kernel

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (dof locs)
    """
    return w.nu * dot(grad(u), grad(v)) + dot(w.vel_f, grad(u)) * v

@fem.BilinearForm
def adv_diff_cons(u, v, w):
    """
    Combo Adv Diff kernel conservative form
    """
    return w.nu * dot(grad(u), grad(v)) - dot(w.vel_f * u, grad(v))

@fem.BilinearForm
def adv_cons(u, v, w):
    """
    Adv kernel, conservative form
    """
    return - dot(w.vel_f * u, grad(v))

@fem.BilinearForm
def diff(u, v, w):
    """
    Diffusion kernel
    """
    return w.nu * dot(grad(u), grad(v))

@fem.BilinearForm
def robin(u, v, w):
    """
    Args:
        w: is a dict of skfem.element.DiscreteField (or user types)
            w.n (face normals)
    """
    return dot(w.vel_f, w.n) * u * v

@fem.LinearForm
def rhs(v, w):
    """
    Source term
    """
    return w.src_f * v

@fem.BilinearForm
def mass(u, v, _):
    return u * v


class AdDiffSEM:
    """
    Assemble matrices for spectral finite element discretization
    of an advection diffusion eqaution

    p: polynomial order
    nu: diffusion coeff
    """
    def __init__(self, mesh, p=1, field_fns={}, params={}, **kwargs):
        # diffusion coeff
        self.params = {"nu": params.get("nu", 5e-3)}
        self.p = p
        self.mesh = mesh

        # register custom field functions
        self.field_fns = {}
        for k, v in field_fns.items():
            self.validate_add_field_fn(k, v)

        if self.p < 2:
            self.element = fem.ElementLineP1()
        elif self.p == 2:
            self.element = fem.ElementLineP2()
        elif self.p >= 3:
            self.element = el_nodal.ElementLinePp_nodal(self.p)

        quad = el_nodal.GLL_quad()(self.p + 1)
        self.basis = fem.Basis(self.mesh, self.element, quadrature=quad)
        self.basis_f = fem.FacetBasis(self.mesh, self.element)

    def validate_add_field_fn(self, field_name: str, f: Callable):
        assert callable(f)
        self.field_fns[field_name] = f

    def w_ext(self, basis, reshape=True, **kwargs):
        """
        Extra kwargs passed with the `w` dict to kernels
        """
        fields = {}
        basis_shape = basis.global_coordinates().shape
        for field_name, field_f in self.field_fns.items():
            vals = field_f(basis.doflocs, **kwargs).flatten()
            # NOTE: reshape is needed here
            # when using these fields in the kernels.
            # is this an issue in skfem??
            fv = basis.interpolate(vals)
            if reshape:
                fv = np.reshape(fv, basis_shape)
            fields[field_name] = fv
        return {**self.params, **fields}

    def assemble(self, **kwargs):
        """
        kwargs: extra args passed to user defined field functions
        """
        conservative = True
        if conservative:
            # remark: w_dict must contain only scalars and skfem.element.DiscreteField
            A_pre = adv_diff_cons.assemble(self.basis, **self.w_ext(self.basis, **kwargs)) \
                    + robin.assemble(self.basis_f, **self.w_ext(self.basis_f, **kwargs))
        else:
            A_pre = adv_diff.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        b_pre = rhs.assemble(self.basis, **self.w_ext(self.basis, reshape=False, **kwargs))
        M_pre = mass.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        # Dirichlet boundary conditions
        A, b = fem.enforce(A_pre, b_pre, D=self.mesh.boundaries['left'].flatten())
        M = fem.enforce(M_pre, D=self.mesh.boundaries['left'].flatten())

        Ml = (M @ np.ones([M.shape[1], 1])).reshape(-1)

        # provide jax arrays
        jMl = jnp.asarray(Ml)
        jA = jsp.BCOO.from_scipy_sparse(A)
        jb = jnp.asarray(b)
        return jA, jMl, jb

    def ode_sys(self, **kwargs):
        return AffineLinearSEM(self, **kwargs)


class AffineLinearSEM(OdeSys):
    """
    Define ODE System associated to affine linear sparse Jacobian problem
    """
    A: jax.Array
    Ml: jax.Array
    b: jax.Array
    jac_lop: Callable
    sys_assembler: AdDiffSEM

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        self.sys_assembler = sys_assembler
        self.A, self.Ml, self.b = self.sys_assembler.assemble(**kwargs)
        self.jac_lop = JaxMatrixLinop(-self.A / self.Ml)
        super().__init__()

    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        # NOTE: in principle one could do:
        # self.A, self.Ml, self.b = self.sys_assembler.assemble(t=t, u=u, **kwargs)
        # here to rebuild the system given the current system state, u
        return (self.b - self.A @ u) / self.Ml

    def _fjac(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        return self.jac_lop


def src_f(x, **kwargs):
    """
    Custom source term, could depend on solution y
    """
    return 0.0 * np.exp(- np.sum((x - 0.2) / 0.1**2, axis=0)**2 / 2)


def vel_f(x, **kwargs):
    """
    Custom velocity field
    """
    return 0.0*x + kwargs.get("vel", 1.0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, gauss]", type=str, default="gauss")
    args = parser.parse_args()

    # create the mesh
    mesh0 = fem.MeshLine1().with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], 1.)
    })
    # mesh refinement
    nrefs=6
    mesh = mesh0.refined(nrefs)

    # diffusion coeff
    nu = 5e-5

    # Specify velocity fn(x)
    vel = 0.5

    # Specify src tem fn(x)
    # src_f = lambda x: 0.0 * np.exp(- np.sum((x - 0.2) / 0.1**2, axis=0)**2 / 2)

    field_fns = {"vel_f": vel_f, "src_f": src_f}
    param_dict = {"nu": nu}

    # init the system
    sem = AdDiffSEM(mesh, p=1, field_fns=field_fns, params=param_dict)
    # ode_sys = sem.ode_sys(vel=vel)
    ode_sys = AffineLinearSEM(sem, vel=vel)
    t = 0.0

    # mesh mask for initial conditions
    x = sem.mesh.doflocs
    sx = np.asarray(x.flatten())

    if args.ic == "square":
        # square wave
        ic_points = np.where((sx > 0.1) & (sx < 0.4))
        y0_profile = np.zeros(sem.jb.shape) + 1e-9
        y0_profile[ic_points] = 1.0
        y0 = jnp.asarray(y0_profile)
    else:
        # gaussian profile
        wc, ww = 0.3, 0.05
        g_prof = lambda x: np.exp(-((x-wc)/(2*ww))**2.0)
        y0_profile = g_prof(sx)
        y0 = jnp.asarray(y0_profile)

        # expected solution
        pass

    # init the time integrator
    sys_int = EpirkIntegrator(ode_sys, t, y0, method="epirk3", max_krylov_dim=20, iom=2)

    t_res = [0,]
    y_res = [y0,]
    dt = .1
    nsteps = 10
    for i in range(nsteps):
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)

    print(np.asarray(y_res))
    # plot the result at a few time steps
    outdir = './advection_diffusion_1d_out/'
    # sorted x
    x = sem.mesh.doflocs
    sx = np.asarray(x.flatten())
    si = sx.argsort()
    sx = sx[si]
    plt.figure()
    for i in range(nsteps):
        if i % 1 == 0 or i == 0:
            t = t_res[i]
            y = y_res[i][si]
            plt.plot(sx, y, label='t=%0.4f' % t)
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.savefig('adv_diff_1d.png')
    plt.close()

    # CFL
    mesh_spacing = (sx[1] - sx[0])
    cfl = dt * vel / mesh_spacing
    print("CFL: %0.4f" % cfl)

