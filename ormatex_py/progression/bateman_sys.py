"""
Methods to define a Bateman system of equations
"""
from typing import Dict, List
import jax
import numpy as np
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from ormatex_py.ode_sys import OdeSys
from ormatex_py.ode_epirk import EpirkIntegrator


decay_lib_0 = {
    'c_0':  ('none', 3.0),
    'c_1':  ('none', 0.3),
    'c_2':  ('none', 0.03),
}

decay_lib_test = {
    'c_0':  ('none', 1.0e-3),
    'c_1':  ('c_0', 1.0e1),
    'c_2':  ('c_1', 1.0e-1),
}

# dict of decay constants
decay_lib_1 = {
    'c_0':  ('none', 3.0),
    'c_1':  ('none', 0.3),
    'c_2':  ('none', 0.03),
    'te_135': ('i_135', jnp.log(2)/19.0),
    'i_135':  ('xe_135', jnp.log(2)/(6.57*3600) ),
    'xe_135': ('cs_135', jnp.log(2) / (9.14*3600) ),
    'cs_135': ('none', jnp.log(2) / (1.33e6*365*24*3600) ),
}

def gen_bateman_matrix(keymap: List, bateman_lib: Dict) -> jax.Array:
    keydict = dict([(k, i) for i, k in enumerate(keymap)])
    d = len(keymap)
    bat_mat = np.zeros((d, d), dtype=np.float64)
    for i, key in enumerate(keymap):
        # lambda = ln(2)/T_1/2 where T_1/2 if the half life in s
        decay_const = bateman_lib[key][1]
        child_species = bateman_lib[key][0]
        if child_species == 'none':
            dest = i
        else:
            dest = keydict[child_species]
            dest = int(dest)
            bat_mat[dest,i] = decay_const
        bat_mat[i,i] = -decay_const
    return jnp.asarray(bat_mat)


class TestBatemanSysFdJac(OdeSys):
    """
    Test fallback to finite diff based Jacobian Linop
    """
    def __init__(self, *args, **kwargs):
        keymap = ["c_0", "c_1", "c_2"]
        bmat = gen_bateman_matrix(keymap, decay_lib_test)
        print("Bateman test system to solve:")
        print(bmat)
        self.keymap = keymap
        self.bat_mat = bmat
        super().__init__()

    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        # print(self.bat_mat)
        # print(u)
        return self.bat_mat @ u


if __name__ == "__main__":
    # associates names with variable index
    keymap = ["c_0", "c_1", "c_2"]
    bmat = gen_bateman_matrix(keymap, decay_lib_0)
    print("diag bateman sys")
    print(bmat)

    keymap = ["c_0", "c_1", "c_2", "te_135", "i_135", "xe_135", "cs_135"]
    bmat = gen_bateman_matrix(keymap, decay_lib_1)
    print("small bateman sys")
    print(bmat)

    # test simple exp integrator
    test_ode_sys = TestBatemanSysFdJac()
    t = 0.0
    y0 = jnp.array([0.001, 0.1, 1.0])
    sys_int = EpirkIntegrator(test_ode_sys, t, y0, 2, method="epirk2", max_krylov_dim=4, iom=5)

    t_res = []
    y_res = []
    dt = 5.0
    nsteps = 10
    for i in range(nsteps):
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)

    for i in range(nsteps):
        print("%0.4e, %0.4e, %0.4e, %0.4e" % (t_res[i], y_res[i][0], y_res[i][1],y_res[i][2]))
