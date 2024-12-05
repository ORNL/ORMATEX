"""
This example solves a system of nine advected, coupled reactive species.
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
from ormatex_py.progression.bateman_sys import gen_bateman_matrix, gen_transmute_matrix
from ormatex_py import integrate_wrapper

# Setup species for tracking
keymap = [
    'c_0_a',
    'c_1_a',
    'c_2_a',
    'te_135_a',
    'i_135_a',
    'xe_135_a',
    'cs_135_a',
    'xe_135_v',
    'cs_135_v',]
gas_purify_lambda = jnp.log(2) / 5.0
decay_lib = {
    # species dissolved in the liquid phase
    'c_0_a':  ('none', 3.0),
    'c_1_a':  ('none', 0.3),
    'c_2_a':  ('none', 0.03),
    'te_135_a': ('i_135_a', jnp.log(2)/19.0),
    'i_135_a':  ('xe_135_a', jnp.log(2)/(6.57*3600) ),
    'xe_135_a': ('cs_135_a', jnp.log(2) / (9.14*3600) ),
    'cs_135_a': ('none', jnp.log(2) / (1.33e6*365*24*3600) ),
    # vapor species
    'xe_135_v': (('cs_135_v', jnp.log(2) / (9.14*3600)),),
    'cs_135_v': ('none', jnp.log(2) / (1.33e6*365*24*3600)),
}

# placeholder absorption cross section of xe_135
sigma_a_xe = 1.0e-3
transmute_lib = {
    'c_0_a':  ('none', 0.0),
    'c_1_a':  ('none', 0.0),
    'c_2_a':  ('none', 0.0),
    'te_135_a': ('none', 0.0),
    'i_135_a':  ('none', 0.0),
    'xe_135_a': ('none', -sigma_a_xe ),  # loss from neutron absorption
    'cs_135_a': ('none', 0.0 ),
    # vapor species
    'xe_135_v': ('none', -sigma_a_xe ),
    'cs_135_v': ('none', 0.0 ),
}
# Transmutation removes or transfers one species to another due to neutron interactions
transmute_mat = gen_transmute_matrix(keymap, transmute_lib)

# placeholder macro fission cross section
sigma_f = 1e-4
fission_lib = {
    'c_0_a':  ('none', 0.03*sigma_f),  # fission_yeild*Sigma_f
    'c_1_a':  ('none', 0.02*sigma_f),
    'c_2_a':  ('none', 0.01*sigma_f),
    'te_135_a': ('none', 0.20*sigma_f),
    'i_135_a':  ('none', 0.01*sigma_f),
    'xe_135_a': ('none', 0.0 ),
    'cs_135_a': ('none', 0.0 ),
    # vapor species
    'xe_135_v': ('none', 0.0 ),
    'cs_135_v': ('none', 0.0 ),
}
# Source terms from fission appear on diagonal of the matrix
fission_mat = gen_transmute_matrix(keymap, fission_lib)
fission_vec = fission_mat.diagonal()


def flux_profile(h: float, x: jax.Array, idxs: jax.Array):
    """
    Neutron flux profile used to drive fission reactions.
    """
    # flux scale
    p = 1.0
    phi = jnp.zeros(x.shape)
    phi = phi.at[idxs[0]].set(jnp.cos(jnp.pi * (x[idxs[0]]-h/2.) / h))
    return jnp.array([p * phi]).transpose()

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
    region_rx: jax.Array

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        # get stiffness matrix and mass vector
        self.A, self.Ml, _ = sys_assembler.assemble(**kwargs)
        # get collocation points
        self.xs = sys_assembler.collocation_points()
        # get bateman matrix
        self.bat_mat = gen_bateman_matrix(keymap, decay_lib)
        # reactor height and center
        h, c = 0.5, 0.25
        self.region_rx = jnp.where( ((c - h/2.) <= self.xs) & (self.xs <= (c + h/2.)) )
        super().__init__()

    def _fission_src(self, un):
        # models production of species from fission only in the reactor
        # region of the mesh
        phi = flux_profile(0.5, self.xs, self.region_rx)
        s = phi * (jnp.ones(un.shape) @ fission_mat.transpose())
        return s

    def _transmute(self, un):
        # models sink of species and species transmutation from neutron capture
        # Xe-135 is rapidly converted to stable Xe-136 by neutron capture
        phi = flux_profile(0.5, self.xs, self.region_rx)
        s = phi * (un @ transmute_mat.transpose())
        return s

    def _nonlin_evap(self, un):
        # models phase transfer for Xe and Cs species from the liquid to vapor phase
        s = jnp.zeros(un.shape)
        # high volotile species
        xe_transfer = mxf_liq_vapor_nonlin(un.at[:,5].get(), un.at[:,7].get(), 4.0e2, 1.0, 1.0)
        s = s.at[:,5].add(xe_transfer)
        s = s.at[:,7].add(-xe_transfer)
        # low volotile species
        cs_transfer = mxf_liq_vapor_nonlin(un.at[:,6].get(), un.at[:,8].get(), 0.2e2, 1.0, 1.0)
        s = s.at[:,6].add(cs_transfer)
        s = s.at[:,8].add(-cs_transfer)
        return s

    def _gas_purify(self, un):
        s = jnp.zeros(un.shape)
        # models off-gas purification where some Xe and Cs vapor are removed
        # in a specific region of the mesh
        # offgass system width
        w = 0.1  # [m]
        # offgas system center
        c = 0.8
        lc, rc = c - w/2., c + w/2.
        # removal efficiency per meter
        eff = 0.01 * w
        offgas_profile = (jnp.tanh((self.xs-lc)*50.0) * (-jnp.tanh((self.xs-rc)*50.0)) ) / 2. + 0.5
        xe_r = eff * un[:, 7] * offgas_profile
        cs_r = eff * un[:, 8] * offgas_profile
        s = s.at[:, 7].add(-xe_r)
        s = s.at[:, 8].add(-cs_r)
        return s

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        n = self.bat_mat.shape[0]
        un = stack_u(u, n)
        # fission
        s_fis = self._fission_src(un)
        # transmute
        s_trans = self._transmute(un)
        # nonlin phase transfer
        s_evap = self._nonlin_evap(un)
        # removal
        s_rem = self._gas_purify(un)
        # bateman
        lub = un @ self.bat_mat.transpose()
        # full system
        udot = lub + (s_fis + s_trans + s_evap + s_rem) - (self.A @ un) / self.Ml.reshape((-1, 1))
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
    parser.add_argument("-nojit", help="Disable jax jit", default=False, action='store_true')
    args = parser.parse_args()
    jax.config.update("jax_disable_jit", args.nojit)

    # create the mesh
    dwidth = 1.0
    mesh0 = fem.MeshLine1(np.array([[0., dwidth]])).with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], dwidth)
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
    param_dict = {"nu": 1e-4, "vel": vel}

    # init the system
    n_species = 9
    sem = AdDiffSEM(mesh, p=args.p, params=param_dict)
    ode_sys = RAD_SEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    # initial profiles for each species
    g_prof0 = lambda x: 0.0*x + 1e-12
    g_prof1 = lambda x: 0.0*x + 1.
    y0_profile = [
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
            g_prof0(xs),
    ]
    y0 = flatten_u(jnp.asarray(y0_profile).transpose())

    # integrate the system
    t0 = 0.
    dt = 0.5
    nsteps = 100
    method = args.method
    t_res, y_res = integrate_wrapper.integrate(ode_sys, y0, t0, dt, nsteps, method, max_krylov_dim=200, iom=10)

    si = xs.argsort()
    sx = xs[si]
    fig, ax = plt.subplots(nrows=9, ncols=1, figsize=(5,20))
    plt.title(r"method: %s" % (method))
    for i in range(nsteps):
        if i == nsteps-1:
            t = t_res[i]
            yf = y_res[i]
            uf = stack_u(yf, n_species)
            for n in range(0, n_species):
                us = uf[:, n]
                ax[n].plot(sx, us[si], label='t=%0.4f, species=%s' % (t, str(n)))
                ax[n].legend()
                ax[n].grid(ls='--')
    # TODO: mark reactor boundaries on the plot
    # ax[1].vlines([0, 0.5], 0.0, 1.0, ls='--', colors='k')
    # ax[0].set_yscale('log')
    ax[0].set_ylabel("concentration [mol/cc]")
    ax[0].set_xlabel("location [m]")
    plt.tight_layout()
    plt.savefig('reac_adv_diff_s9.png')
    plt.close()
