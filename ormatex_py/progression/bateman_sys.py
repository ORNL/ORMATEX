"""
Methods to define a Bateman system of equations
"""
import jax
import numpy as np
from jax import numpy as jnp

from ormatex_py import integrate_wrapper
from ormatex_py.ode_sys import OdeSys, OdeSplitSys, MatrixLinOp
from ormatex_py.ode_exp import ExpRBIntegrator

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False


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
    'te_135': ('i_135', np.log(2.) / 19.0),
    'i_135':  ('xe_135', np.log(2.) / (6.57*3600) ),
    'xe_135': ('cs_135', np.log(2.) / (9.14*3600) ),
    'cs_135': ('none', np.log(2.) / (1.33e6*365*24*3600) ),
}

def gen_bateman_matrix(keymap: list, bateman_lib: dict) -> jax.Array:
    r"""
    Represents nuclear decay chain reactions of the form:

    The diagonal is given by decay constants of each species:

    .. math::

        T_{i,j} = -\lambda_{i}, i == j

    The lower triangle is given by positive entries representing
    decay from a parent species into another child species.

    .. math::

        T_{i,j} = b_{i,j}*\lambda_{i,j}, i > j

    .. math::

        u_t = T u

    Where $`\lambda_i`$ is the decay constant of species i, and
    $`\lambda_{i,j}`$ is the decay constant associated with the decay
    branch with factor, $`b_{i,j}`$.
    """
    keydict = dict([(k, i) for i, k in enumerate(keymap)])
    d = len(keymap)
    bat_mat = np.zeros((d, d), dtype=np.float64)
    for i, key in enumerate(keymap):
        if isinstance(bateman_lib[key][0], tuple):
            for child_lambda_pair in bateman_lib[key]:
                child_species = child_lambda_pair[0]
                decay_const = child_lambda_pair[1]
                if child_species == 'none':
                    dest = i
                else:
                    dest = keydict[child_species]
                    dest = int(dest)
                    bat_mat[dest,i] += decay_const
                bat_mat[i,i] -= decay_const
        else:
            # lambda = ln(2)/T_1/2 where T_1/2 if the half life in s
            child_species = bateman_lib[key][0]
            decay_const = bateman_lib[key][1]
            if child_species == 'none':
                dest = i
            else:
                dest = keydict[child_species]
                dest = int(dest)
                bat_mat[dest,i] += decay_const
            bat_mat[i,i] -= decay_const
    return jnp.asarray(bat_mat)


def gen_transmute_matrix(keymap: list, trans_lib: dict, phi: float=1.0) -> jax.Array:
    r"""
    Represents transmutation reactions of the form:

    .. math::

        T_{i,j} = \Sigma_{i,j}

    .. math::

        u_t = \phi T u

    Where $`\phi`$ is the neutron scalar flux and $`\Sigma_{i,j}`$ is the
    macroscopic cross section of reaction that transmutes species i to j.
    This yeilds a matrix of similar form to the Bateman decay matrix.

    This is purely a convinience function and to denote that the
    transmuation lib is not equal to the bateman lib since
    the transmutation lib contains $`\Sigma_{i,j}`$ values.
    """
    return -phi * gen_bateman_matrix(keymap, trans_lib)


def analytic_bateman_single_parent(t, batmat, n0):
    r"""
    Helper function to quickly evaluate analytic solution
    to a small bateman system.

    This routine only works when all species concentrations
    are initially zero, except for one parent species.  In
    this special case, the solution to the Bateman equations is
    given by

    .. math::

        N_i(t) = N_0(0) \left( \prod_{j=0}^{i-1}\lambda_i \right) \sum_{k=0}^i \frac{e^{-\lambda_k t}}{\prod_{l=0,l \ne k} \lambda_l - \lambda_k}

    This result was derived by:
        Bateman, H. The solution of a system of differential equations
        occurring in the theory of radioactive transformations.
        In Proc. Cambridge Philos. Soc. Vol. 15. 1910.
        https://en.wikipedia.org/wiki/Bateman_equation

    Note:  This routine does not work if two adjacent decay
    constants in the decay chain are nearly equal.  This is due to
    $` \lambda_l - lambda_k `$ in the denomenator.  Under most
    normal processes, this is not a concern.

    Args:
        t: array of times to evaluate analytic results at
        batmat: bateman matrix
        n0: initial concentration of the 0th species
    """
    blib = np.asarray(batmat, dtype=np.float64)
    ns = blib.shape[0]
    nt = len(t)
    assert blib.shape[0] == blib.shape[1]
    # check that diagonal is equal to neg subdiagonal
    diag = blib.diagonal()
    sdiag = blib.diagonal(offset=-1)
    assert np.allclose(sdiag, -1.0*diag[0:ns-1])
    lmbd = -1.0*diag
    N = np.zeros((nt, ns), dtype=np.float64)
    for n in range(1, ns+1):
        lmbd_prod = 1.0
        for i in range(n-1):
            if i >= 0:
                lmbd_prod *= lmbd[i]
        sum_k = np.zeros(nt, dtype=np.float64)
        for i in range(n):
            prod_l = 1.0
            for j in range(n):
                if j == i:
                    continue
                prod_l *= (lmbd[j] - lmbd[i])
            sum_k += np.exp(-lmbd[i]*t)/prod_l
        N[:, n-1] = n0 * lmbd_prod * sum_k
    return np.asarray(N, dtype=np.float64)


def analytic_bateman_s3(method="epi2", do_plot=True, dt=10.0, tf=1000.):
    jax.config.update("jax_enable_x64", True)
    keymap = ["c_0", "c_1", "c_2"]
    decay_lib_sp = {
        'c_0':  ('c_1', 1.0e-1),
        'c_1':  ('c_2', 1.0e1),
        'c_2':  ('none', 1.0e-3),
    }
    bmat = gen_bateman_matrix(keymap, decay_lib_sp)
    n0 = 1.0
    t0 = 0.0
    t = np.arange(t0, tf+dt, dt)
    # analytic result
    y_true = analytic_bateman_single_parent(t, bmat, n0)

    # compute numerical result
    test_ode_sys = TestBatemanSysJac(keymap, decay_lib_sp)
    y0 = jnp.array([n0, 0.0, 0.0])
    nsteps = int((tf - t0) / dt)
    res = integrate_wrapper.integrate(test_ode_sys, y0, t0, dt, nsteps, method, max_krylov_dim=12, iom=12)
    t_res, y_res = res.t_res, res.y_res
    t_res = np.asarray(t_res)
    y_res = np.asarray(y_res)

    # plot
    if do_plot and HAS_MATPLOTLIB:
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim((1e-8, 10.0))
        plt.xlim((1.0, tf))
        # numerical
        plt.plot(t_res+1.0, y_res[:, 0]+1e-16, label="c_0")
        plt.plot(t_res+1.0, y_res[:, 1]+1e-16, label="c_1")
        plt.plot(t_res+1.0, y_res[:, 2]+1e-16, label="c_2")
        # analytic
        plt.plot(t+1.0, y_true[:, 0]+1e-16, ls='--', label="c_0 true")
        plt.plot(t+1.0, y_true[:, 1]+1e-16, ls='--', label="c_1 true")
        plt.plot(t+1.0, y_true[:, 2]+1e-16, ls='--', label="c_2 true")
        plt.legend()
        plt.grid(ls='--')
        plt.title(r"Method: %s, $\Delta t=$%0.2f" % (method, dt))
        plt.ylabel("Species concentration")
        plt.xlabel("Time [s]")
        plt.savefig("bateman_analytic_3s_%s_dt_%0.2f.png" % (method, dt))
        plt.close()

    return t_res, y_res, t, y_true


def run_sweep():
    methods = ["epi2", "epi3", "exprb3", "exp2_dense", "exp3_dense",
               "exprb2_dense", "exprb2_dense_cauchy", "dense_cauchy",
               "implicit_euler", "implicit_esdirk3", "implicit_esdirk4"]
    dts = [1., 2., 5., 10., 25., 50.]
    tf = 100.
    nspecies = 3
    mae_dict = {}
    for method in methods:
        err_arr = np.zeros((len(dts), nspecies+1))
        for j, dt in enumerate(dts):
            t_res, y_res, t, y_true = \
                    analytic_bateman_s3(method, dt=dt, tf=tf, do_plot=False)
            diff = y_res - y_true
            err_arr[j, 0] = dt
            # loop over species at last time
            for s in range(diff.shape[1]):
                rel_diff = np.abs(diff[-1, s]) / y_true[-1, s]
                err_arr[j, s+1] = rel_diff + 1e-18
        mae_dict[method] = err_arr
        print("=== Method: %s" % method)
        print("dt, err_s0, err_s1, err_s2")
        print(err_arr)

    # error vs time step size for each method
    plt.figure()
    for method in methods:
        dt = mae_dict[method][:, 0]
        s3_err = mae_dict[method][:, 2]
        plt.plot(dt, s3_err, '-o', label=method)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(ls='--')
    plt.title("$t_f$=%0.2f (s)" % tf)
    plt.xlabel(r"Time step size $\Delta$t (s)")
    plt.ylabel(r"Species Rel. Err |calc-true| / true")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bateman_ex_1_converg.png")
    for method in methods:
        for dt in [10., 25.]:
            analytic_bateman_s3(method, dt=dt, tf=500., do_plot=True)


class TestBatemanSysJac(OdeSplitSys):
    """
    Test fallback to finite diff based Jacobian LinOp
    """
    bat_mat: jax.Array

    def __init__(self, keymap, decay_lib, *args, **kwargs):
        bmat = gen_bateman_matrix(keymap, decay_lib)
        print("Bateman test system to solve:")
        print(bmat)
        self.bat_mat = bmat
        super().__init__()

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        res = self.bat_mat @ u
        return res

    # define the Jacobian LinOp (comment out to use autograd)
    def _fjac(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        return MatrixLinOp(self.bat_mat)

    def _fl(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        return MatrixLinOp(self.bat_mat)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    parser.add_argument("-sweep", help="run convergence sweep", action="store_true", default=False)
    args = parser.parse_args()
    method = args.method

    # associates names with variable index
    keymap = ["c_0", "c_1", "c_2"]
    bmat = gen_bateman_matrix(keymap, decay_lib_test)
    print("diag bateman sys")
    print(bmat)

    # test simple exp integrator
    test_ode_sys = TestBatemanSysJac(keymap, decay_lib_test)
    t = 0.0
    y0 = jnp.array([0.001, 0.1, 1.0])

    # step system forward
    t0 = 0.0
    tf = 1000.0
    dt = 10.0
    nsteps = int((tf - t0) / dt)
    res = integrate_wrapper.integrate(test_ode_sys, y0, t0, dt, nsteps, method, max_krylov_dim=12, iom=12)
    t_res, y_res = res.t_res, res.y_res

    t_res = np.asarray(t_res)
    y_res = np.asarray(y_res)
    for i in range(nsteps):
        print("%0.4e, %0.4e, %0.4e, %0.4e" % (t_res[i], y_res[i][0], y_res[i][1],y_res[i][2]))

    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((1e-14, 10.0))
    plt.plot(t_res, y_res[:, 0], label="c_0")
    plt.plot(t_res, y_res[:, 1], label="c_1")
    plt.plot(t_res, y_res[:, 2], label="c_2")
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel("Species concentration")
    plt.xlabel("Time [s]")
    plt.savefig("bateman_ex_1_%s.png" % method)
    plt.close()

    if args.sweep:
        run_sweep()
