"""
Regression test for non linear Lotka-Volterra system.

Check that all exponential integrators solve a simple
nonlinear system to the expected precission.
"""
from ormatex_py import integrate_wrapper
from ormatex_py.progression.lotka_volterra import LotkaVolterra
import numpy as np
import scipy as sp
import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)


def test_lotka_volterra():
    """
    Test exponential integrators on the Lotka-Volterra system
    """
    sys = LotkaVolterra()
    dt_list = [0.01, 0.0125, 0.02, 0.05]
    methods = ["epi2", "epi3", "exprb3"]
    methods_order = [2.0, 3.0, 3.0]
    for method, order in zip(methods, methods_order):
        err_dt = []
        for dt in dt_list:
            y0 = jnp.array([0.1, 0.2])
            # compute expected result
            t0 = 0.0
            tf = 20.0
            nsteps = int(tf/dt)
            res_true = integrate_wrapper.integrate(
                    sys, y0, t0, dt, nsteps, "rk4")
            t_true, y_true = res_true.t_res, res_true.y_res

            # compute ormatex result
            y0 = jnp.array([0.1, 0.2])
            res = integrate_wrapper.integrate(
                    sys, y0, t0, dt, nsteps, method, max_krylov_dim=10, iom=5)
            t_res, y_res = res.t_res, res.y_res
            t_res = np.asarray(t_res)
            y_res = np.asarray(y_res)
            diff = y_res - y_true
            mae = np.mean(np.abs(diff))
            print("Method: %s, dt: %0.4e, mean abs err: %0.4e" % (method, dt, mae))
            assert mae < 1e-2
            err_dt.append([dt, mae])
        err_dt = np.array(err_dt)

        # estimate conv order
        trendf = lambda x, s, b: s*np.log(x)+b
        popt, _ = sp.optimize.curve_fit(trendf, err_dt[:, 0], np.log(err_dt[:, 1]), p0=[1.0, 1.0])
        est_order = popt[0]
        print("Method: %s, Est conv order: %0.4e" % (method, est_order))
        order_tol = 1e-1
        assert est_order - order > (0.0 - order_tol)
