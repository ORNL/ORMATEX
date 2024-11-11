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

def src_f(x, **kwargs):
    """
    Custom source term, could depend on solution y

    Args:
        x: Array of shape [1,N1,N2,...] (the first dim is the spatial dim=1)

    returns: Array of shape [N1,N2,...]
    """
    return 0.0 * x.sum(axis=0)

def vel_f(x, vel, **kwargs):
    """
    Custom velocity field

    Args:
        x: Array of shape [1,N1,N2,...] (the first dim is the spatial dim=1)
        vel: velocity coefficient

    returns: Array of shape x.shape
    """
    return 0.0 * x + vel

@fem.BilinearForm
def adv_diff(u, v, w):
    """
    Combo Adv Diff kernel

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (quadrature points)
    """
    return w['nu'] * dot(grad(u), grad(v)) + dot(vel_f(**w), grad(u)) * v

@fem.BilinearForm
def adv_diff_cons(u, v, w):
    """
    Combo Adv Diff kernel conservative form

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (quadrature points)
    """
    return w['nu'] * dot(grad(u), grad(v)) - dot(vel_f(**w) * u, grad(v))

@fem.BilinearForm
def robin(u, v, w):
    """
    Args:
        w: is a dict of skfem.element.DiscreteField (or user types)
            w.n (face normals)
    """
    return dot(vel_f(**w), w.n) * u * v

@fem.LinearForm
def rhs(v, w):
    """
    Source term
    """
    return src_f(**w) * v

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
        self.params = {"nu": params.get("nu", 5e-3), "vel": params.get("vel", 1.)}
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
        if self.mesh.boundaries:
            # if the mesh has boundaries, get a basis for BC
            self.basis_f = fem.FacetBasis(self.mesh, self.element)

    def validate_add_field_fn(self, field_name: str, f: Callable):
        assert callable(f)
        self.field_fns[field_name] = f

    def w_ext(self, basis, **kwargs):
        """
        Extra kwargs passed with the `w` dict to kernels
        """
        fields = {}
        for field_name, field_f in self.field_fns.items():
            # nodal field values
            fx = field_f(x=basis.doflocs, **self.params, **kwargs)

            # turn vector fields into scalar fields for interpolate (flatten in 1D)
            # basis is the basis for a scalar field and can not handle vector fields
            fv = basis.interpolate(fx.flatten())

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
            if self.mesh.boundaries:
                # if not periodic, add boundary term
                A += robin.assemble(self.basis_f, **self.w_ext(self.basis_f, **kwargs))
        else:
            A = adv_diff.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        b = rhs.assemble(self.basis, **self.w_ext(self.basis, **kwargs))
        M = mass.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        if self.mesh.boundaries:
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

    def _frhs(self, t: float, u: jax.Array) -> jax.Array:
        # NOTE: in principle one could do:
        # self.A, self.Ml, self.b = self.sys_assembler.assemble(t=t, u=u, **kwargs)
        # here to rebuild the system given the current system state, u
        return (self.b - self.A @ u) / self.Ml

    @property
    def jac_lop(self):
        return JaxMatrixLinop(-self.A / self.Ml[:,None])

    def _fjac(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        return self.jac_lop

    def _fm(self, t: float, u: jax.Array, **kwargs):
        return lambda x: self.Ml * x


def integrate_diffrax(ode_sys, y0, dt, nsteps, method="implicit_euler"):
    """
    Uses diffrax integrators to step adv diff system forward
    """
    import diffrax
    import optimistix
    # thin wrapper around ode_sys for diffrax compat
    diffrax_ode_sys = diffrax.ODETerm(ode_sys)
    method_dict = {
            # explicit
            "euler": diffrax.Euler,
            "heun": diffrax.Heun,
            "midpoint": diffrax.Midpoint,
            "bosh3": diffrax.Bosh3,
            "dopri5": diffrax.Dopri5,
            # implicit
            "implicit_euler": diffrax.ImplicitEuler,
            "implicit_esdirk3": diffrax.Kvaerno3,
            "implicit_esdirk4": diffrax.Kvaerno4,
           }
    try:
        root_finder=diffrax.VeryChord(rtol=1e-8, atol=1e-8, norm=optimistix.max_norm)
        solver = method_dict[method](root_finder=root_finder)
    except:
        solver = method_dict[method]()
    t0 = 0.0
    tf = dt * nsteps
    step_ctrl = diffrax.ConstantStepSize()
    res = diffrax.diffeqsolve(
            diffrax_ode_sys,
            solver,
            t0, tf, dt, y0,
            saveat=diffrax.SaveAt(steps=True),
            stepsize_controller=step_ctrl,
            max_steps=nsteps,
            )
    return res.ts, res.ys


def integrate_ormatex(ode_sys, y0, dt, nsteps, method="epirk3", max_krylov_dim=40, iom=2):
    """
    Uses ormatex exponential integrators to step adv diff system forward
    """
    # init the time integrator
    t0 = 0.0
    sys_int = EpirkIntegrator(
            ode_sys, t0, y0, method=method,
            max_krylov_dim=max_krylov_dim, iom=iom
            )
    t_res, y_res = [0,], [y0,]
    for i in range(nsteps):
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)
    return t_res, y_res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, gauss]", type=str, default="gauss")
    parser.add_argument("-mr", help="mesh refinement", type=int, default=6)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-method", help="time step method", type=str, default="epirk3")
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

    # overall velocity vel
    vel = 0.5
    # diffusion coefficient nu
    nu = 0.0
    ##nu = 0.05 * vel / (args.p * 2**nrefs) #stabilization by diffusion
    param_dict = {"nu": nu, "vel": vel}
    field_dict = {} #{"vel_f": vel_f, "src_f": src_f}

    # init the system
    sem = AdDiffSEM(mesh, p=args.p, params=param_dict, field_fns=field_dict)
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
        g_prof_exact = lambda t, x: np.exp(-( ( (x - (wc+t*vel)%1 ) ) /(2*ww))**2.0)

    # integrate the system
    dt = .1
    nsteps = 10
    method = args.method
    diffrax_methods = ["euler", "heun", "midpoint", "bosh3", "dopri5", "implicit_euler", "implicit_esdirk3", "implicit_esdirk4"]
    if method in diffrax_methods:
        t_res, y_res = integrate_diffrax(ode_sys, y0, dt, nsteps, method=method)
    else:
        t_res, y_res = integrate_ormatex(ode_sys, y0, dt, nsteps, method=method)

    # compute expected solution
    # y_exact_res = [g_prof_exact(0.0, sx),]
    y_exact_res = []
    for t in t_res:
        y_exact_res.append(g_prof_exact(t, xs))

    # plot the result at a few time steps
    #outdir = './advection_diffusion_1d_out/'
    # sorted x
    si = xs.argsort()
    sx = xs[si]
    mesh_spacing = (sx[1] - sx[0])
    cfl = dt * vel / mesh_spacing
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
    plt.title(r"Method=%s, $C$=%0.2e, $\Delta$ t=%0.2e, $\Delta$ x=%0.2e" % (method, cfl, dt, mesh_spacing))
    plt.savefig('adv_diff_1d.png')
    plt.close()

    # Print results summary to table
    print("CFL: %0.4f" % cfl)

    err = y_exact - y
    l2 = np.sqrt(np.sum(err**2 * ode_sys.Ml))
    l1 = np.linalg.norm(err * ode_sys.Ml, 1)
    linf = np.linalg.norm(err, np.inf)
    print("mesh_spacing: %0.4e, CFL=%0.4f, L1=%0.4e, L2=%0.4e, Linf=%0.4e" % (mesh_spacing, cfl, l1, l2, linf))
