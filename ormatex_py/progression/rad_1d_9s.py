"""
This example solves a system of nine advected, coupled reactive species.
This example contains nonlinear reactions, and
coupled to the advection and diffusion system,
no analytic solution is availible.

The purpose of this example is to demonstrate the relative
performance of different numerical integration schemes and
to compare the results of the exponential integrators to
traditional implicit time integration methods.
"""
import numpy as np
import scipy as sp
import os

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
import equinox as eqx

import skfem as fem

try:
    from ormatex_py.ormatex import PySysWrapped
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False
from ormatex_py.ode_sys import OdeSys, OdeSplitSys, OdeSysNp, MatrixLinOp
from ormatex_py.ode_utils import stack_u, flatten_u
from ormatex_py.progression.rad_1d_3s import plot_dt_jac_spec, plot_leja_conv_detail
from ormatex_py.progression.species_source_sink import mxf_liq_vapor_bubble_ig, mxf_arrhenius, mxf_liq_vapor_nonlin
from ormatex_py.progression.advection_diffusion_1d import AdDiffSEM
from ormatex_py.progression.bateman_sys import gen_bateman_matrix, gen_transmute_matrix
from ormatex_py import integrate_wrapper

outdir = "./rad_1d_9s_out/"
if not os.path.exists(outdir): os.mkdir(outdir)

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
gas_purify_lambda = np.log(2) / 5.0
decay_lib = {
    # species dissolved in the liquid phase
    'c_0_a':  ('none', 3.0),
    'c_1_a':  ('none', 0.3),
    'c_2_a':  ('none', 0.03),
    'te_135_a': ('i_135_a', np.log(2)/19.0),
    'i_135_a':  ('xe_135_a', np.log(2)/(6.57*3600) ),
    'xe_135_a': ('cs_135_a', np.log(2) / (9.14*3600) ),
    'cs_135_a': ('none', np.log(2) / (1.33e6*365*24*3600) ),
    # vapor species
    'xe_135_v': (('cs_135_v', np.log(2) / (9.14*3600)),),
    'cs_135_v': ('none', np.log(2) / (1.33e6*365*24*3600)),
}

# placeholder absorption cross section of xe_135
sigma_a_xe = 9.0e-3
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
    'c_0_a':  ('none', 0.01*sigma_f),  # fission_yeild*Sigma_f
    'c_1_a':  ('none', 0.03*sigma_f),
    'c_2_a':  ('none', 0.02*sigma_f),
    'te_135_a': ('none', 0.18*sigma_f),
    'i_135_a':  ('none', 0.15*sigma_f),
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

class RAD_SEM(OdeSplitSys):
    """
    Define ODE System associated to RAD problem
    """
    bat_mat: jax.Array
    A: jsp.JAXSparse
    Ml: jax.Array
    xs: jax.Array
    region_rx: jax.Array

    dirichlet_bd: np.array

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        # get stiffness matrix and mass vector
        self.A, self.Ml, _ = sys_assembler.assemble(**kwargs)
        # get collocation points
        self.xs = sys_assembler.collocation_points()

        # Dirichlet boundary conditions
        if sys_assembler.dirichlet_bd is not None:
            self.dirichlet_bd = sys_assembler.dirichlet_bd
        else:
            self.dirichlet_bd = np.array([], dtype=int)

        # get bateman matrix
        self.bat_mat = gen_bateman_matrix(keymap, decay_lib)
        # reactor height and center
        h, c = 1.0, 0.5
        self.region_rx = jnp.where( ((c - h/2.) <= self.xs) & (self.xs <= (c + h/2.)) )
        super().__init__()

    def _fission_src(self, un):
        # models production of species from fission only in the reactor
        # region of the mesh
        phi = flux_profile(1.0, self.xs, self.region_rx)
        s = phi * (jnp.ones(un.shape) @ fission_mat.transpose())
        return s

    def _transmute(self, un):
        # models sink of species and species transmutation from neutron capture
        # Xe-135 is rapidly converted to stable Xe-136 by neutron capture
        phi = flux_profile(1.0, self.xs, self.region_rx)
        s = phi * (un @ transmute_mat.transpose())
        return s

    def _nonlin_evap(self, un):
        # models phase transfer for Xe and Cs species from the liquid to vapor phase
        s = jnp.zeros(un.shape)
        # high volotile species
        xe_transfer = mxf_liq_vapor_nonlin(un.at[:,5].get(), un.at[:,7].get(), 10.0e3, 1.0, 1.0)
        s = s.at[:,5].add(xe_transfer)
        s = s.at[:,7].add(-xe_transfer)
        # low volotile species
        cs_transfer = mxf_liq_vapor_nonlin(un.at[:,6].get(), un.at[:,8].get(), 0.9e3, 1.0, 1.0)
        s = s.at[:,6].add(cs_transfer)
        s = s.at[:,8].add(-cs_transfer)
        return s

    def _gas_purify(self, un):
        s = jnp.zeros(un.shape)
        # models off-gas purification where some Xe and Cs vapor are removed
        # in a specific region of the mesh
        # offgass system width
        w = 0.4  # [m]
        # offgas system center
        c = 4.0
        lc, rc = c - w/2., c + w/2.
        # removal efficiency per meter
        eff = 0.40 * w
        offgas_profile = (jnp.tanh((self.xs-lc)*10.0) * (-jnp.tanh((self.xs-rc)*10.0)) ) / 2. + 0.5
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
        # set the time derivative of the Dirichlet boundary data to zero
        udot = udot.at[self.dirichlet_bd,:].set(0.)
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
    parser.add_argument("-nsteps", help="number of time steps", type=int, default=100)
    parser.add_argument("-multi_plot", help="multiple plots", action='store_true', default=False)
    parser.add_argument("-per", help="impose periodic BC", action='store_true')
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    parser.add_argument("-fine", help="compare to fine step solution", default=False, action='store_true')
    parser.add_argument("-leja_c", help="optional max complex part of the J*dt spectrum", type=float, default=20.0)
    parser.add_argument("-leja_plot", help="additional leja polynomial convergence plots", default=False, action='store_true')
    parser.add_argument("-dt", help="time step size", type=float, default=1.0)
    parser.add_argument("-nojit", help="Disable jax jit", default=False, action='store_true')
    args = parser.parse_args()
    jax.config.update("jax_disable_jit", args.nojit)

    # create the mesh
    dwidth = 5.0
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
    param_dict = {"nu": 5e-4, "vel": vel}

    # init the system
    n_species = 9
    sem = AdDiffSEM(mesh, p=args.p, params=param_dict)
    ode_sys = RAD_SEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    # initial profiles for each species
    g_prof0 = lambda x: 0.0*x + 1e-16
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
    dt = args.dt
    nsteps = args.nsteps
    tf = dt * nsteps
    method = args.method
    if args.fine:
        # Compute ground-truth baseline solution
        dt_fine = 0.5
        nsteps_fine = int(tf / dt_fine)
        res_fine = integrate_wrapper.integrate(
                ode_sys, y0, t0, dt_fine, nsteps_fine, "exprb3",
                max_krylov_dim=240, iom=2)
        t_res_fine, y_res_fine = res_fine.t_res, res_fine.y_res
    if "_rs" in method:
        # use a rust ormatex integrator
        # NOTE: the rust integrators currently require
        # converting jax types to np, so the wrapper does this
        # conversion automatically but with some overhead.
        # Despite this, on a multi-core CPU, the rust exp int
        # impls are slightly faster than the JAX impl.
        y0 = np.asarray(y0).reshape((-1, 1))
        res = integrate_wrapper.integrate(
                PySysWrapped(OdeSysNp(ode_sys)), y0, t0, dt, nsteps,
                method, max_krylov_dim=200, iom=2, osteps=20)
        t_res, y_res = res.t_res, res.y_res
    else:
        # use a python ormatex integrator
        res = integrate_wrapper.integrate(
                ode_sys, y0, t0, dt, nsteps, method,
                max_krylov_dim=200, iom=2, leja_c=args.leja_c, leja_tol=1e-12)
        t_res, y_res = res.t_res, res.y_res

    si = xs.argsort()
    sx = xs[si]
    print("Mesh Spacing: %0.4e" % (sx[2] - sx[0]))
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,8.5))
    plt.title(r"method: %s" % (method))

    i = -1
    t = t_res[i]
    yf = y_res[i]
    uf = stack_u(yf, n_species)
    # fine solution
    if args.fine:
        i_fine = nsteps_fine
        t_fine = t_res_fine[i_fine]
        yf_fine = y_res_fine[i_fine]
        uf_fine = stack_u(yf_fine, n_species)

    for n in range(0, n_species):
        us = uf[:, n]
        # fig row,col
        fr, fc = n%3, int(n/3)
        ax[fr, fc].axvspan(0.0, 1.0, alpha=0.3, color='red')
        ax[fr, fc].axvspan(3.8, 4.2, alpha=0.3, color='blue')
        ax[fr, fc].plot(sx, us[si], label=r't=%0.4f, $u_{%s}$' % (t, str(n)))
        if args.fine:
            us_fine = uf_fine[:, n]
            rel_diff = (us[si] - us_fine[si]) / jnp.mean(us_fine)
            mae = jnp.mean(jnp.abs(rel_diff))
            ax[fr, fc].plot(sx, us_fine[si], ls='--', label='t=%0.4f, baseln $u_{%s}$' % (t_fine, str(n)))
            ax[fr, fc].set_title(r"%s, $\Delta t=$%0.2e, Rel.Err=%0.3e" % (method, dt, mae))
        ax[fr, fc].set_ylabel(r"$u_{%d}$ [mol/cc]" % n)
        ax[fr, fc].legend()
        ax[fr, fc].grid(ls='--')

    # TODO: mark reactor boundaries on the plot
    # ax[1].vlines([0, 0.5], 0.0, 1.0, ls='--', colors='k')
    # ax[0].set_yscale('log')
    # ax[0].set_xlabel("location [m]")
    plt.tight_layout()
    plt.savefig(outdir + 'reac_adv_diff_s9.png', dpi=160)
    plt.close()

    if args.multi_plot:
        # plot results at multiple time steps
        for i in range(0, len(t_res)):
            fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,8.5))
            plt.title(r"method: %s" % (method))
            t = t_res[i]
            yf = y_res[i]
            uf = stack_u(yf, n_species)
            # fine solution
            if args.fine:
                i_fine = nsteps_fine
                t_fine = t_res_fine[i_fine]
                yf_fine = y_res_fine[i_fine]
                uf_fine = stack_u(yf_fine, n_species)

            for n in range(0, n_species):
                us = uf[:, n]
                # fig row,col
                fr, fc = n%3, int(n/3)
                ax[fr, fc].axvspan(0.0, 1.0, alpha=0.3, color='red')
                ax[fr, fc].axvspan(3.8, 4.2, alpha=0.3, color='blue')
                ax[fr, fc].plot(sx, us[si], label=r't=%0.4f, $u_{%s}$' % (t, str(n)))
                if args.fine:
                    us_fine = uf_fine[:, n]
                    rel_diff = (us[si] - us_fine[si]) / jnp.mean(us_fine)
                    mae = jnp.mean(jnp.abs(rel_diff))
                    ax[fr, fc].plot(sx, us_fine[si], ls='--', label='t=%0.4f, baseln $u_{%s}$' % (t_fine, str(n)))
                    ax[fr, fc].set_title(r"%s, $\Delta t=$%0.2e, Rel.Err=%0.3e" % (method, dt, mae))
                ax[fr, fc].set_ylabel(r"$u_{%d}$ [mol/cc]" % n)
                ax[fr, fc].legend()
                ax[fr, fc].grid(ls='--')

            plt.tight_layout()
            plt.savefig(outdir + 'reac_adv_diff_s12_%d.png' % i, dpi=160)
            plt.close()

    # plot eigvals of Jac
    plot_dt_jac_spec(ode_sys, y_res[-1], t=0., dt=dt, figname=outdir + "reac_adv_diff_s9_eigplot")

    if "leja" in method and args.leja_plot:
        kwargs = {"leja_c": args.leja_c}
        plot_leja_conv_detail(ode_sys, y_res[-1], t=0., dt=dt, outdir=outdir, **kwargs)
