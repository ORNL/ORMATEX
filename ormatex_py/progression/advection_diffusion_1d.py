import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp

import skfem as fem
from skfem.helpers import dot, grad
import warnings

from ormatex_py.progression import element_line_pp_nodal as el_nodal

jax.config.update("jax_enable_x64", True)

from ormatex_py.ode_sys import OdeSys
from ormatex_py.ode_epirk import EpirkIntegrator


## problem and weak form specification

nu = 5e-3
vel = lambda x: (x+1.)/(x+1.)
f = lambda x: np.exp(- np.sum((x - 0.2) / 0.1**2, axis=0)**2 / 2)

@fem.BilinearForm
def adv_diff(u, v, w):
    return nu * dot(grad(u), grad(v)) + dot(vel(w.x), grad(u)) * v

@fem.BilinearForm
def adv_diff_cons(u, v, w):
    return nu * dot(grad(u), grad(v)) - dot(vel(w.x) * u, grad(v))

@fem.BilinearForm
def robin(u, v, w):
    return dot(vel(w.x), w.n) * u * v

@fem.LinearForm
def rhs(v, w):
    return f(w.x) * v

@fem.BilinearForm
def mass(u, v, _):
    return u * v

class AdDiffSEM:
    """
    Assemble matrices for spectral finite element discretization of an advection diffusion eqaution

    p: polynomial order
    nrefs: refinement level of the mesh
    """
    def __init__(self, p=1, nrefs=1):
        self.p = p
        self.nrefs = nrefs
        self.assemble()

    def assemble(self):

        # create the mesh
        mesh0 = fem.MeshLine1().with_boundaries({
            'left': lambda x: np.isclose(x[0], 0.),
            'right': lambda x: np.isclose(x[0], 1.)
        })
        self.mesh = mesh0.refined(self.nrefs)

        if self.p < 2:
            self.element = fem.ElementLineP1()
        elif self.p == 2:
            self.element = fem.ElementLineP2()
        elif self.p >= 3:
            self.element = el_nodal.ElementLinePp_nodal(self.p)
            ## implementation with skfem provided element does not support easy mass lumping for p >= 3
            #self.element = fem.ElementLinePp(self.p)

        #if self.p < 3:
        #    quad = GL_quad[self.p + 1]
        #    basis = fem.Basis(self.mesh, self.element, quadrature=quad)
        #else:
        #    warnings.warn(f"Dofs are not Lagrange basis for p={self.p}. Mass lumping not applied.")
        #    basis = fem.Basis(self.mesh, self.element)

        quad = el_nodal.GLL_quad()(self.p + 1)
        basis = fem.Basis(self.mesh, self.element, quadrature=quad)
        basis_f = fem.FacetBasis(self.mesh, self.element)

        conservative = True
        if conservative:
            self.A_pre = adv_diff_cons.assemble(basis) + robin.assemble(basis_f)
        else:
            self.A_pre = adv_diff.assemble(basis)

        self.b_pre = rhs.assemble(basis)
        self.M_pre = mass.assemble(basis)

        # Dirichlet boundary conditions
        self.A, self.b = fem.enforce(self.A_pre, self.b_pre, D=self.mesh.boundaries['left'].flatten())
        self.M = fem.enforce(self.M_pre, D=self.mesh.boundaries['left'].flatten())

        self.Ml = (self.M @ np.ones([self.M.shape[1], 1])).reshape(-1)

        # provide jax arrays
        self.jMl = jnp.asarray(self.Ml)
        self.jA = jsp.BCOO.from_scipy_sparse(self.A)
        self.jb = jnp.asarray(self.b)

    def get_initial(self):
        # return jnp.zeros(self.jb.shape)
        return jnp.ones(self.jb.shape)

    def ode_sys(self):
        return AffineLinearSEM(self.jA, self.jMl, self.jb)


class AffineLinearSEM(OdeSys):
    """
    Define ODE System associated to affine linear sparse Jacobian problem
    """
    def __init__(self, A: jsp.JAXSparse, Ml: jax.Array, b: jax.Array, *args, **kwargs):
        self.A = A
        self.Ml = Ml
        self.b = b
        super().__init__()

    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        return (self.b - self.A @ u) / self.Ml


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    jax.config.update('jax_platform_name', 'cpu')
    print(f"Running on {jax.devices()}.")

    # test simple exp integrator
    sem = AdDiffSEM(p=2, nrefs=4)
    ode_sys = sem.ode_sys()
    t = 0.0

    # mesh mask for bcs
    x = sem.mesh.doflocs
    sx = np.asarray(x.flatten())
    ic_points = np.where((sx > 0.1) & (sx < 0.4))

    # square wave
    y0_profile = np.zeros(sem.jb.shape) + 1e-9
    y0_profile[ic_points] = 1.0
    y0 = jnp.asarray(y0_profile)

    sys_int = EpirkIntegrator(ode_sys, t, y0, method="epirk2")

    t_res = [0,]
    y_res = [y0,]
    dt = .01
    nsteps = 100
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
        if i % 10 == 0 or i == 0:
            t = t_res[i]
            y = y_res[i][si]
            plt.plot(sx, y, label='t=%0.4f' % t)
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.savefig('adv_diff_1d.png')
    plt.close()

