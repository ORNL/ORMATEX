"""
Multiple species 1D reaction advection diffusion case.

- Supports N numbers of coupled species.
- Each species can have unique, space, time, and concentration
  dependant non-linear sources and sinks.

Similar to the single species case (advection_diffusion_1d.py);
however, internally the species vector, `u` is now a 2D tensor,
with each column representing a new species.

All species are assumed to be advected according to the same
velocity field.  All species share the same bulk diffusion coefficient.

We can write:
    U_t = L*u + v*\nabla U - D \nabla^2 U + S(U)

where

    U = {u_0, u_1, ..., u_N}

    or in matrix form

U = [
     [u_{0,0}, ... u_{N,0},],
     [u_{0,1},     u_{N,1} ],
     [...      ...         ],
     [u_{0,X}  ... u_{N,X} ],
]
"""
import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
import equinox as eqx

import skfem as fem

from ormatex_py.ode_sys import OdeSys, OdeSplitSys, MatrixLinOp

from ormatex_py.progression.species_source_sink import mxf_liq_vapor_bubble_ig, mxf_arrhenius, mxf_liq_vapor_nonlin
from ormatex_py.progression.advection_diffusion_1d import AdDiffSEM
from ormatex_py.progression.bateman_sys import gen_bateman_matrix
from ormatex_py import integrate_wrapper

# decay lib
decay_lib = {
    # more stiff:'u_0':  ('u_1', 3e1),
    'u_0':  ('u_1', 3.e-1),
    'u_1':  ('none', .3e-1),
}

def stack_u(u: jax.Array, n: int):
    return u.reshape((-1, n), order='F')

def flatten_u(u: jax.Array):
    # use column major ordering
    return u.reshape((-1, ), order='F')

class RAD_SEM(OdeSplitSys):
    """
    Define ODE System associated to RAD problem
    """
    bat_mat: jax.Array
    A: jsp.JAXSparse
    Ml: jax.Array
    xs: jax.Array

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        # get stiffness matrix and mass vector
        self.A, self.Ml, _ = sys_assembler.assemble(**kwargs)
        # get collocation points
        self.xs = sys_assembler.collocation_points()
        # get bateman matrix
        self.bat_mat = gen_bateman_matrix(['u_0', 'u_1'], decay_lib)
        super().__init__()

    def _source_poly(self, un):
        #n = un.shape[1]
        s = jnp.zeros(un.shape)
        # implement a nonlinear transfer from last to first species
        transfer = 10. * un[:,-1]**5
        s = s.at[:,0].add(transfer)
        s = s.at[:,1].add(-transfer)
        #TODO implement more reasonable / interesting nonlinear source
        return s

    def _source_nonlin_evap(self, un):
        s = jnp.zeros(un.shape)
        transfer = mxf_liq_vapor_nonlin(un.at[:,0].get(), un.at[:-1].get(), 1e-4, 1.0, 1.0)
        import pdb; pdb.set_trace()
        s = s.at[:,0].add(transfer)
        s = s.at[:,1].add(-transfer)
        return s

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        n = self.bat_mat.shape[0]
        un = stack_u(u, n)
        # nonlin rhs
        s = self._source_nonlin_evap(un)
        # add bateman
        lub = un @ self.bat_mat.transpose()
        udot = lub + s - (self.A @ un) / self.Ml.reshape((-1, 1))
        # integrators currently expect a flat U
        return flatten_u(udot)

    @jax.jit
    def _fl(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        """
        define a linear operator (for testing purposes)
        # TODO: uses dense Kronecker product, should be a BlockDiagLinOp (not implemented yet)
        """
        n = self.bat_mat.shape[0]
        Ndof = self.Ml.shape[0]
        # only use Bateman terms for L
        #L = jnp.kron(self.bat_mat, jnp.eye(Ndof))
        # only use adv-diff terms for L
        L = jnp.kron(jnp.eye(n), (- self.A / self.Ml.reshape((-1, 1))).todense())
        #print(L)
        return MatrixLinOp(L)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    jax.config.update("jax_enable_x64", True)
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, gauss]", type=str, default="gauss")
    parser.add_argument("-mr", help="mesh refinement", type=int, default=6)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-per", help="impose periodic BC", action='store_true')
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
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

    # diffusion coefficient
    vel = 0.5
    param_dict = {"nu": 1e-8, "vel": vel}

    # init the system
    n_species = 2
    sem = AdDiffSEM(mesh, p=args.p, params=param_dict)
    ode_sys = RAD_SEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    # initial profiles for each species
    wc, ww = 0.3, 0.05
    g_prof = lambda x: np.exp(-((x-wc)/(2*ww))**2.0)
    g_prof2 = lambda x: 0.2*np.exp(-((x-wc)/(2*ww))**2.0)
    y0_profile = [g_prof(xs), g_prof2(xs)]
    y0 = flatten_u(jnp.asarray(y0_profile).transpose())

    # integrate the system
    t0 = 0.
    dt = .1
    nsteps = 20
    method = args.method
    t_res, y_res = integrate_wrapper.integrate(ode_sys, y0, t0, dt, nsteps, method, max_krylov_dim=200, iom=10)

    si = xs.argsort()
    sx = xs[si]
    plt.figure()
    for i in range(nsteps):
        if i % 2 == 0 or i == 0:
            t = t_res[i]
            yf = y_res[i]
            uf = stack_u(yf, n_species)
            for n in range(n_species):
                us = uf[:, n]
                line_style = None
                if (n+1) % 2 == 0:
                    line_style = '--'
                plt.plot(sx, us[si], ls=line_style, label='t=%0.4f, species=%s' % (t, str(n)))
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.title(r"method: %s" % (method))
    plt.savefig('reac_adv_diff_1d.png')
    plt.close()
