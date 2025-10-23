import numpy as np
import scipy as sp

from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp

from collections.abc import Callable

import skfem as fem
from skfem.helpers import dot, grad, div, dd

from ormatex_py.progression import element_line_pp_nodal as el_nodal
from ormatex_py.ode_sys import LinOp, MatrixLinOp, DiagLinOp

from ormatex_py.ode_sys import OdeSys, OdeSplitSys
from ormatex_py.ode_exp import ExpRBIntegrator

from ormatex_py import integrate_wrapper

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

def tau_upwind_f(w):
    """
    Computes the SUPG stabilization parameter using the standard
    upwind formulation.
    """
    return w['tau'] * w.h[:] / (2 * vel_f(**w))

@fem.BilinearForm
def adv_diff(u, v, w):
    """
    Combo Adv Diff kernel in non-conservative form:

        u_t = k nabla(u) + vel*grad(u)

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (quadrature points)
    """
    return w['nu'] * dot(grad(u), grad(v)) + dot(vel_f(**w), grad(u)) * v

@fem.BilinearForm
def adv_diff_cons(u, v, w):
    """
    Combo Adv Diff kernel conservative form:

        u_t = k nabla(u) + grad(vel * u)

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (quadrature points)
    """
    return w['nu'] * dot(grad(u), grad(v)) - dot(vel_f(**w) * u, grad(v))

@fem.BilinearForm
def adv_diff_cons_supg(u, v, w):
    """
    Combo Adv Diff kernel conservative form with SUPG stab.

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (quadrature points)
    """
    r = w['nu'] * dot(grad(u), grad(v)) - dot(vel_f(**w) * u, grad(v))
    # advection supg term (adds artificial diffusion)
    tau = tau_upwind_f(w)
    stab_adv = dot(tau * vel_f(**w), grad(v)) * dot(vel_f(**w), grad(u))
    # diffusion supg term
    # NOTE: second deriv of u is not defined everywhere for linear basis fns. so this fails.
    # stab_diff = dot(w['tau'] * vel_f(**w), grad(v)) * w['nu'] * div(grad(u))
    return r + stab_adv

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

@fem.LinearForm
def rhs_supg(v, w):
    """
    Source term
    """
    r = src_f(**w) * v
    tau = tau_upwind_f(w)
    stab_f =  dot(tau * vel_f(**w), grad(v)) * src_f(**w)
    return r - stab_f

@fem.BilinearForm
def mass(u, v, _):
    return u * v

@fem.BilinearForm
def mass_supg(u, v, w):
    r =  u * v
    tau = tau_upwind_f(w)
    stab_m = dot(tau * vel_f(**w), grad(v)) * u
    return r + stab_m


class AdDiffSEM:
    """
    Assemble matrices for spectral finite element discretization
    of an advection diffusion eqaution

    p: polynomial basis order
    nu: physical diffusion coeff
    tau: supg stabilization scale. 0 for no stabilization
    """
    def __init__(self, mesh, p=1, field_fns={}, params={}, **kwargs):
        self.params = {
                "nu": params.get("nu", 5e-3),
                "vel": params.get("vel", 1.),
                "tau": params.get("tau", 0.0),
                }
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

        self.dirichlet_bd = None
        if self.mesh.boundaries:
            # set the left boundary to Dirichlet
            self.dirichlet_bd = self.mesh.boundaries['left'].flatten()
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
            A = adv_diff_cons_supg.assemble(self.basis, **self.w_ext(self.basis, **kwargs))
            if self.mesh.boundaries:
                # if not periodic, add boundary term
                A += robin.assemble(self.basis_f, **self.w_ext(self.basis_f, **kwargs))
        else:
            A = adv_diff.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        b = rhs_supg.assemble(self.basis, **self.w_ext(self.basis, **kwargs))
        M = mass_supg.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        if self.mesh.boundaries:
            # Dirichlet boundary conditions
            fem.enforce(A, b, D=self.dirichlet_bd, overwrite=True)
            fem.enforce(M, D=self.dirichlet_bd, overwrite=True)
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

    def collocation_points(self):
        return jnp.asarray(self.basis.doflocs[0])

    def ode_sys(self, **kwargs):
        return AffineLinearSEM(self, **kwargs)


class AffineLinearSEM(OdeSplitSys):
    """
    Define ODE System associated to affine linear sparse Jacobian problem

    The system is affine linear and the Jacobian is constant in t an u.
    Implement a linear operator L equal to Jacobian to test non-Rosenbrock schemes.
    All schemes should be exact (up to Krylov error) since the residual term vanishes.
    """
    A: jsp.JAXSparse
    Ml: jax.Array
    b: jax.Array

    dirichlet_bd: np.array

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        self.A, self.Ml, self.b = sys_assembler.assemble(**kwargs)

        # Dirichlet boundary conditions
        if sys_assembler.dirichlet_bd is not None:
            self.dirichlet_bd = sys_assembler.dirichlet_bd
        else:
            self.dirichlet_bd = np.array([], dtype=int)

        super().__init__()

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        f = (self.b - self.A @ u) / self.Ml
        # set the time derivative of the Dirichlet boundary data to zero
        f = f.at[self.dirichlet_bd].set(0.)
        return f

    @jax.jit
    def _fl(self, t: float, u: jax.Array, **kwargs):
        return MatrixLinOp(-self.A / self.Ml[:,None])

    def _fm(self, t: float, u: jax.Array, **kwargs):
        return DiagLinOp(self.Ml)


class NonautonomousSEM(AffineLinearSEM):
    """
    Define ODE System associated to affine linear sparse Jacobian problem

    The same as AffinLinearSEM, but with nonautonomous Dirichlet boundary conditions
    """

    @jax.jit
    def dirichlet_dt(self, t: float):
        # use the dirichlet data corresponding to default analytic solution
        wc, ww = 0.3, 0.05
        vel = 0.5
        dirichlet_fun = \
            lambda time: jnp.exp(-(torus_distance(0., (wc+time*vel))/(2*ww))**2.0)
        # return the time derivative of the Dirichlet data
        return jax.grad(dirichlet_fun)(t)

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        f = (self.b - self.A @ u) / self.Ml
        # set the time derivative of the Dirichlet boundary data
        f = f.at[self.dirichlet_bd].set(self.dirichlet_dt(t))
        return f

def torus_distance(x, xp):
    """ distance of two points on torus (up to equivalence) """
    dx = jnp.abs(x%1 - xp%1)
    return jnp.where(dx > 0.5, 1. - dx, dx)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    jax.config.update("jax_enable_x64", True)
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, zero, gauss]", type=str, default="gauss")
    parser.add_argument("-mr", help="mesh refinement", type=int, default=6)
    parser.add_argument("-tau", help="supg stabilization constant. 0 for no supg stab.", type=float, default=0.0)
    parser.add_argument("-nu", help="physical species bulk diffusion coefficient. 0 for no diffusion.", type=float, default=1.0e-12)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    parser.add_argument("-pfd_method", help="partial frac decomp method", type=str, default="CN")
    parser.add_argument("-nsteps", help="number of steps", type=int, default=10)
    parser.add_argument("-per", help="impose periodic BC", action='store_true')
    parser.add_argument("-tf", help="final time", type=float, default=1.6)
    parser.add_argument("-leja_c", help="leja scale", type=float, default=10.0)
    parser.add_argument("-dd_method", help="divided difference method", type=str, default="taylor")
    parser.add_argument("-nonautonomous", help="run nonautonomous system with external forcing", action="store_true", default=False)
    args = parser.parse_args()

    # create the mesh
    mesh0 = fem.MeshLine1().with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], 1.)
    })
    # mesh refinement
    nrefs = args.mr
    mesh = mesh0.refined(nrefs)

    print(mesh)
    periodic = args.per
    if periodic:
        mesh = fem.MeshLine1DG.periodic(
            mesh,
            mesh.boundaries['right'],
            mesh.boundaries['left'],
        )

    # overall velocity vel and diffusion coefficient nu
    if args.ic == "zero":
        vel = 0.1
        nu =  1.0
    else:
        vel = 0.5
        nu =  args.nu
        # nu = 0.05 * vel / (args.p * 2**nrefs) #stabilization by diffusion
    param_dict = {"nu": nu, "vel": vel, "tau": args.tau}
    field_dict = {} #{"vel_f": vel_f, "src_f": src_f}

    # init the system
    sem = AdDiffSEM(mesh, p=args.p, params=param_dict, field_fns=field_dict)
    if not args.nonautonomous:
        ode_sys = AffineLinearSEM(sem)
    else:
        ode_sys = NonautonomousSEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.collocation_points())

    dist = lambda x, xp: jnp.abs(x - xp)
    if periodic or args.nonautonomous:
        # distance on the torus
        dist = lambda x, xp: torus_distance(x, xp)

    if args.ic == "square":
        # square wave
        startx, endx = 0.1, 0.4
        meanx, dxhalf = (endx + startx)/2., (endx - startx)/2.
        g_prof = lambda x: np.where(dist(meanx, x) < dxhalf, 1., 0.)
        y0_profile = g_prof(xs)
        y0 = jnp.asarray(y0_profile)
        g_prof_exact = lambda t, x: g_prof(x - t*vel)
    elif args.ic == "zero":
        g_prof = lambda x: np.zeros(x.shape)
        y0_profile = g_prof(xs)
        y0 = jnp.asarray(y0_profile)
        g_prof_exact = lambda t, x: g_prof(x - t*vel)
    else:
        # gaussian profile
        wc, ww = 0.3, 0.05
        g_prof = lambda x: np.exp(-(dist(x, wc) / (2*ww))**2.0)
        y0_profile = g_prof(xs)
        y0 = jnp.asarray(y0_profile)
        g_prof_exact = lambda t, x: g_prof(x - t*vel)

    # modification for Dirichlet boundary conditions
    if sem.dirichlet_bd is not None:
        if args.ic == "zero":
            y_dir = 1
        else:
            y_dir = g_prof(np.array([0.]))
        y0 = y0.at[sem.dirichlet_bd].set(y_dir)

    # integrate the system
    t0 = 0.
    nsteps = args.nsteps
    dt = args.tf / nsteps
    method = args.method
    pfd_method = args.pfd_method
    res = integrate_wrapper.integrate(
            ode_sys, y0, t0, dt, nsteps, method,
            max_krylov_dim=800, iom=2, pfd_method=pfd_method,
            leja_c=args.leja_c, dd_method=args.dd_method)
    t_res, y_res = res.t_res, res.y_res

    # compute expected solution
    y_exact_res = []
    for t in t_res:
        y_exact_res.append(g_prof_exact(t, xs))

    # plot the result at a few time steps
    # sorted x
    si = xs.argsort()
    sx = xs[si]
    mesh_spacing = (sx[1] - sx[0])
    cfl = dt * vel / mesh_spacing
    plt.figure()
    for i in range(nsteps+1):
        if i == nsteps or i == 0:
            t = t_res[i]
            y = y_res[i][si]
            y_exact = y_exact_res[i][si]
            plt.plot(sx, y, label='t=%0.4f' % t)
            plt.plot(sx, y_exact, ls='--', label='exact t=%0.4f' % t)
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.title("Method=%s, $C$=%0.2e, $v_{adv.}$=%0.2e \n $tau$=%0.2e, $\Delta$ t=%0.2e, $\Delta$ x=%0.2e" % (method, cfl, vel, args.tau, dt, mesh_spacing))
    plt.savefig('adv_diff_1d_%s_%s_%s.png' % (method, str(args.mr), str(args.ic)))
    plt.close()

    # Print results summary to table
    print("CFL: %0.4f, Ndof: %d" % (cfl, xs.size))

    err = y_exact - y
    l2 = np.sqrt(np.sum(err**2 * ode_sys.Ml))
    l1 = np.linalg.norm(err * ode_sys.Ml, 1)
    linf = np.linalg.norm(err, np.inf)
    print("mesh_spacing: %0.4e, CFL=%0.4f, L1=%0.4e, L2=%0.4e, Linf=%0.4e" % (mesh_spacing, cfl, l1, l2, linf))
