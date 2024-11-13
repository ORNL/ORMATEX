"""
Nonlinear example progression problems
"""
import jax
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import diffrax
jax.config.update("jax_enable_x64", True)

from ormatex_py.ode_sys import OdeSplitSys, MatrixLinOp
from ormatex_py.ode_exp import ExpRBIntegrator, ExpSplitIntegrator


class LotkaVolterra(OdeSplitSys):
    alpha: float
    beta: float
    delta: float
    gamma: float

    def __init__(self, **kwargs):
        super().__init__()
        self.alpha = kwargs.get("alpha", 1.0)
        self.beta = kwargs.get("beta", 1.0)
        self.delta = kwargs.get("delta", 1.0)
        self.gamma = kwargs.get("gamma", 1.0)

    def _frhs(self, t, x, **kwargs):
        prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
        pred_t = self.delta * x[0] * x[1] - self.gamma * x[1]
        return jnp.array([prey_t, pred_t])

    # define the Jacobian LinOp (comment out to use autograd)
    def _fjac(self, t, x, **kwargs):
        jac = jnp.array([
            [self.alpha - self.beta * x[1], - self.beta*x[0]],
            [self.delta*x[1], self.delta*x[0] - self.gamma]
            ])
        return MatrixLinOp(jac)

    # define a linear operator for testing
    def _fl(self, t, x, **kwargs):
        lop = jnp.array([
            [self.alpha - self.beta*2, 0.],
            [0., - self.gamma]
            ])
        return MatrixLinOp(lop)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default="exprb2")
    args = parser.parse_args()
    method = args.method

    # setup lotka voltera system
    lv_sys = LotkaVolterra()
    t_end = 20.0
    y0 = jnp.array([0.1, 0.2])

    # compute gold solution
    t0 = 0.0
    dt = 0.001
    nsteps = int(t_end/dt)
    tf = dt * nsteps
    step_ctrl = diffrax.ConstantStepSize()
    solver = diffrax.Dopri5()
    diffrax_lv_sys = diffrax.ODETerm(lv_sys)
    res = diffrax.diffeqsolve(
            diffrax_lv_sys,
            solver,
            t0, tf, dt, y0,
            saveat=diffrax.SaveAt(steps=True),
            stepsize_controller=step_ctrl,
            max_steps=nsteps,
            )
    t_gold, y_gold = res.ts, res.ys

    # sweep over time step size and integrate
    dt_list = [0.0125, 0.025, 0.05, 0.125, 0.25]
    #dt_list = [0.5*dt for dt in dt_list]
    t_list = []
    y_list = []
    for dt in dt_list:
        print("dt = %0.2e" % dt)
        t = 0.0
        try:
            sys_int = ExpRBIntegrator(lv_sys, t, y0, method=method, max_krylov_dim=10, iom=5)
        except AssertionError:
            sys_int = ExpSplitIntegrator(lv_sys, t, y0, method=method, max_krylov_dim=10, iom=5)

        nsteps = int(t_end/dt)
        t_res, y_res = [], []
        for i in range(nsteps):
            res = sys_int.step(dt)
            t_res.append(res.t)
            y_res.append(res.y)
            sys_int.accept_step(res)
        t_res = jnp.asarray(t_res)
        y_res = jnp.asarray(y_res)
        t_list.append(t_res)
        y_list.append(y_res)

        fig, ax1 = plt.subplots()
        ax1.plot(t_res, y_res[:, 0], label="prey")
        ax1.plot(t_res, y_res[:, 1], label="pred")
        ax1.plot(t_gold, y_gold[:, 0], ls='--', label="prey true")
        ax1.plot(t_gold, y_gold[:, 1], ls='--', label="pred true")
        ax1.set(ylim=(-1., 6.))
        plt.legend()
        plt.grid(ls='--')
        ax1.set_xlabel("time")
        ax1.set_ylabel("population")
        # interp the fine gold results to coarse grid
        y_gold_int = jnp.interp(t_res, t_gold, y_gold[:, 0])
        diff = y_res[:, 0] - y_gold_int
        ax2 = ax1.twinx()
        ax2.plot(t_res, diff, c='k', ls='--')
        ax2.set_ylabel("prey diff")
        plt.savefig("lotka_volterra_%s_dt_%s.png" % (str(method), str(dt)))
        plt.close()

    # compare solutions
    err_dt = []
    for dt, t_res, y_res in zip(dt_list, t_list, y_list):
        # final time match check
        assert abs(t_res[-1] - t_gold[-1]) <= 1e-12
        y_gold_int = jnp.interp(t_res, t_gold, y_gold[:, 0])
        diff = y_res[:, 0] - y_gold_int
        err = jnp.mean(jnp.abs(diff))
        err_dt.append((dt, err))
    err_dt = jnp.asarray(err_dt)
    print(err_dt)
    # fit trendline for plot, log(y)
    trendf = lambda x, s, b: s*np.log(x)+b
    popt, _ = sp.optimize.curve_fit(trendf, err_dt[:, 0], np.log(err_dt[:, 1]), p0=[1.0, 1.0])
    print("est conv order: ", popt[0])
    plt.figure()
    plt.scatter(err_dt[:, 0], err_dt[:, 1])
    plt.plot(err_dt[:, 0], np.exp(trendf(err_dt[:, 0], popt[0], popt[1])), ls='--', label="trendline, s=%0.2e" % popt[0])
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"|diff|")
    plt.grid(ls='--')
    plt.title("method: %s" % str(method))
    plt.savefig("lotka_volterra_%s_conv.png" % (str(method)))
    plt.close()
