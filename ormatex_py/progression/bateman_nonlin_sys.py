"""
Bateman system with nonlinear source and sinks
from phase transfer from liquid species
to and from a vapor species.

This models a tank with an immobile mixture of
vapor bubbles and liquid at some volume fraction vapor.
We assume the species transfering to and from
the vapor phase are trace elements and do not impact
the volume fraction of the total vapor.
"""
import matplotlib.pyplot as plt
import jax
import numpy as np
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from ormatex_py.progression import integrate_wrapper
from ormatex_py.progression.bateman_sys import gen_bateman_matrix, gen_transmute_matrix
from ormatex_py.progression.species_source_sink import mxf_liq_vapor_bubble_ig, mxf_arrhenius, mxf_liq_vapor_nonlin

from ormatex_py.ode_sys import OdeSys, MatrixLinOp, OdeSplitSys
from ormatex_py.ode_exp import ExpRBIntegrator

# removal rate constants from gas purification
gas_purify_lambda = jnp.log(2) / 1.0
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
    'xe_135_v': (('cs_135_v', jnp.log(2) / (9.14*3600)), ('none', gas_purify_lambda),),
    'cs_135_v': ('none', jnp.log(2) / (1.33e6*365*24*3600) + gas_purify_lambda),
}
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
bateman_mat = gen_bateman_matrix(keymap, decay_lib)
# NOTE: the only way to get from xe_135_a to xe_135_v is due to phase transfer.
# initially, the concentration of xe_135_v is 0.

# placeholder absorption cross section of xe_135
sigma_a_xe = 1.0e-9
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
sigma_f = 1.0 * 1.0e-9
fission_lib = {
    'c_0_a':  ('none', 0.03*sigma_f),  # fission_yeild*Sigma_f
    'c_1_a':  ('none', 0.02*sigma_f),
    'c_2_a':  ('none', 0.01*sigma_f),
    'te_135_a': ('none', 0.20*sigma_f),
    'i_135_a':  ('none', 0.01*sigma_f),
    'xe_135_a': ('none', 0.0 ),  # loss from neutron absorption
    'cs_135_a': ('none', 0.0 ),
    # vapor species
    'xe_135_v': ('none', 0.0 ),
    'cs_135_v': ('none', 0.0 ),
}
# Source terms from fission appear on diagonal of the matrix
fission_mat = gen_transmute_matrix(keymap, fission_lib)
fission_vec = fission_mat.diagonal()

def srcf_xe_av(u, i_a, i_g, alpha_g=0.05, tank_vol=100.0):
    """
    Liquid to vapor mass transfer for Xe species
    """
    src = jnp.zeros(u.shape)
    u_a = u[i_a]
    u_g = u[i_g]
    s = mxf_liq_vapor_bubble_ig(u_a, u_g, tank_vol, alpha_g, k=1e-2)
    # remove from liquid
    src = src.at[i_a].set(s)
    # add to vapor
    src = src.at[i_g].set(-s)
    return src


# class NonlinearBateman(OdeSys):
class NonlinearBateman(OdeSplitSys):
    # neutron flux in n/cm^2/s
    n_phi: float
    # tank volume in cc
    tank_vol: float
    bat_mat: jax.Array
    trans_mat: jax.Array
    fis_vec: jax.Array
    i_xe_a: int
    i_xe_g: int

    def __init__(self, **kwargs):
        # get index of species used in the nonlinear terms
        self.i_xe_a = keymap.index("xe_135_a")
        self.i_xe_g = keymap.index("xe_135_v")
        self.n_phi = kwargs.get("n_phi", 1.0e9)
        self.tank_vol = kwargs.get("tank_vol", 100.)
        self.bat_mat = bateman_mat
        self.trans_mat = transmute_mat
        self.fis_vec = fission_vec

    def _frhs(self, t, u, **kwargs):
        nonlin = srcf_xe_av(u, self.i_xe_a, self.i_xe_g)
        bat = self.bat_mat @ u
        trans = self.n_phi * (self.trans_mat @ u)
        fis = self.n_phi * self.fis_vec
        u_dot = bat + trans + fis + nonlin
        return u_dot

    def _fl(self, t, u, **kwargs):
        return MatrixLinOp(self.bat_mat)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    args = parser.parse_args()
    method = args.method

    bateman_sys = NonlinearBateman()
    t = 0.0
    # initially, clean salt
    y0 = jnp.asarray([1e-16] * len(keymap))

    # step system forward
    t0 = 0.0
    dt = 5.0
    nsteps = 800
    t_res, y_res = integrate_wrapper.integrate(bateman_sys, y0, t0, dt, nsteps, method, max_krylov_dim=12, iom=2)
    t_res = np.asarray(t_res)
    y_res = np.asarray(y_res)
    print(y_res)

    dense_jac = bateman_sys.fjac(0.0, y_res[-1]).dense()
    print(dense_jac)
    # import pdb; pdb.set_trace()

    plt.figure()
    for n in range(y_res.shape[1]):
        species_name = keymap[n]
        u_n = y_res[:, n]
        plt.plot(t_res, u_n, label="species: %s" % str(species_name))
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim((1e-14, 1e3))
    plt.xlim((1e-1, None))
    plt.grid(ls='--')
    plt.legend()
    plt.ylabel("species concentration (mol/cc)")
    plt.xlabel("time (s)")
    plt.title(r"method: %s, $\Delta$ t: %s" % (str(method), str(dt)))
    plt.tight_layout()
    plt.savefig("bateman_nonlin_n9_dt_%s_%s.png" % (str(dt), str(method)))
    plt.close()
