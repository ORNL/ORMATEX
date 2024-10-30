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

# diffusion coeff
nu = 1e-12

# Specify velocity
vel = 0.5

def src_f(x, **kwargs):
    """
    Custom source term, could depend on solution y
    """
    return 0.0 * np.exp(- np.sum((x - 0.2) / 0.1**2, axis=0)**2 / 2)


def vel_f(x, **kwargs):
    """
    Custom velocity field
    """
    return 0.0*x + vel

@fem.BilinearForm
def adv_diff(u, v, w):
    """
    Combo Adv Diff kernel

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (dof locs)
    """
    return w.nu * dot(grad(u), grad(v)) + dot(vel_f(w.x), grad(u)) * v

@fem.BilinearForm
def adv_diff_cons(u, v, w):
    """
    Combo Adv Diff kernel conservative form
    """
    return w.nu * dot(grad(u), grad(v)) - dot(vel_f(w.x) * u, grad(v))

@fem.BilinearForm
def adv_cons(u, v, w):
    """
    Adv kernel, conservative form
    """
    return - dot(vel_f(w.x) * u, grad(v))

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
    return dot(vel_f(w.x), w.n) * u * v

@fem.LinearForm
def rhs(v, w):
    """
    Source term
    """
    return src_f(w.x) * v

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
        if mesh.boundaries:
            # if the mesh has boundaries, get a basis for BC
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
            A = adv_diff_cons.assemble(self.basis, **self.w_ext(self.basis, **kwargs))
            if mesh.boundaries:
                # if not periodic, add boundary term
                A += robin.assemble(self.basis_f, **self.w_ext(self.basis_f, **kwargs))
        else:
            A = adv_diff.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        b = rhs.assemble(self.basis, **self.w_ext(self.basis, **kwargs))
        M = mass.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        if mesh.boundaries:
            # Dirichlet boundary conditions
            fem.enforce(A, b, D=self.mesh.boundaries['left'].flatten(), overwrite=True)
            fem.enforce(M, D=self.mesh.boundaries['left'].flatten(), overwrite=True)
        else:
            # periodic boundary conditions
            fem.enforce(A, b, D=self.basis.get_dofs().flatten(), overwrite=True)
            fem.enforce(M, D=self.basis.get_dofs().flatten(), overwrite=True)

        Ml = M @ np.ones((M.shape[1],))

        # provide jax arrays
        jMl = jnp.asarray(Ml)
        jA = jsp.BCOO.from_scipy_sparse(A)
        jb = jnp.asarray(b)

        return jA, jMl, jb

    def xs(self):
        return self.basis.doflocs[0]

    def ode_sys(self, **kwargs):
        return AffineLinearSEM(self, **kwargs)


class AffineLinearSEM(OdeSys):
    """
    Define ODE System associated to affine linear sparse Jacobian problem
    """
    A: jsp.JAXSparse
    Ml: jax.Array
    b: jax.Array
    jac_lop: Callable
    sys_assembler: AdDiffSEM

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        self.sys_assembler = sys_assembler
        self.A, self.Ml, self.b = self.sys_assembler.assemble(**kwargs)
        super().__init__()

    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        # NOTE: in principle one could do:
        # self.A, self.Ml, self.b = self.sys_assembler.assemble(t=t, u=u, **kwargs)
        # here to rebuild the system given the current system state, u
        return (self.b - self.A @ u) / self.Ml

    @property
    def jac_lop(self):
        return JaxMatrixLinop(-self.A / self.Ml[:,None])

    def _fjac(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        return self.jac_lop


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, gauss]", type=str, default="gauss")
    parser.add_argument("-mr", help="mesh refinement", type=int, default=6)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-per", help="impose periodic BC", action='store_true') 
    args = parser.parse_args()

    # create the mesh
    mesh0 = fem.MeshLine1().with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], 1.)
    })
    # mesh refinement
    nrefs = args.mr
    mesh = mesh0.refined(nrefs)

    periodic = args.per
    if periodic: 
        mesh = fem.MeshLine1DG.periodic(
            mesh,
            mesh.boundaries['right'],
            mesh.boundaries['left'],
        )

    param_dict = {"nu": nu}

    # init the system
    sem = AdDiffSEM(mesh, p=args.p, params=param_dict)
    ode_sys = AffineLinearSEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    if args.ic == "square":
        # square wave
        startx, endx = 0.1, 0.4
        ic_points = np.where((xs > startx) & (xs < endx))
        y0_profile = np.zeros(ode_sys.b.shape) + 1e-9
        y0_profile[ic_points] = 1.0
        y0 = jnp.asarray(y0_profile)
        g_prof_exact = lambda t, x: \
                np.asarray([1.0 if (x_i > startx+t*vel) & (x_i < endx+t*vel) else 0.0 for x_i in x])
    else:
        # gaussian profile
        wc, ww = 0.3, 0.05
        g_prof = lambda x: np.exp(-((x-wc)/(2*ww))**2.0)
        y0_profile = g_prof(xs)
        y0 = jnp.asarray(y0_profile)

        # expected solution
        g_prof_exact = lambda t, x: np.exp(-((x-(wc+t*vel))/(2*ww))**2.0)

    # init the time integrator
    
    sys_int = EpirkIntegrator(ode_sys, t, y0, method="epirk3", max_krylov_dim=100, iom=2)

    t_res = [0,]
    y_res = [y0,]
    y_exact_res = [g_prof_exact(0.0, xs),]
    dt = .2
    nsteps = 10
    for i in range(nsteps):
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        y_exact_res.append(g_prof_exact(res.t, xs))
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)

    # plot the result at a few time steps
    #outdir = './advection_diffusion_1d_out/'
    # sorted x
    si = xs.argsort()
    sx = xs[si]
    plt.figure()
    for i in range(nsteps):
        if i % 2 == 0 or i == 0:
            t = t_res[i]
            y = y_res[i][si]
            y_exact = y_exact_res[i][si]
            plt.plot(sx, y, label='t=%0.4f' % t)
            plt.plot(sx, y_exact, ls='--', label='exact t=%0.4f' % t)
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.savefig('adv_diff_1d.png')

    # CFL
    mesh_spacing = (sx[1] - sx[0])
    cfl = dt * vel / mesh_spacing
    print("CFL: %0.4f" % cfl)

    err = y_exact - y
    l2 = np.sqrt(np.sum(err**2 * ode_sys.Ml))
    l1 = np.linalg.norm(err * ode_sys.Ml, 1)
    linf = np.linalg.norm(err, np.inf)
    print("mesh_spacing: %0.4e, CFL=%0.4f, L1=%0.4e, L2=%0.4e, Linf=%0.4e" % (mesh_spacing, cfl, l1, l2, linf))

    #plt.show()
    plt.close()
