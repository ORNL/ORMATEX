"""
Nonlinear example progression problems
"""
import jax
import scipy as sp
import numpy as np
from jax import numpy as jnp

from ormatex_py import integrate_wrapper
from ormatex_py.ode_sys import OdeSplitSys, OdeSys, MatrixLinOp, CustomJacLinOp, FdJacLinOp
from ormatex_py.ode_exp import ExpRBIntegrator, ExpSplitIntegrator
try:
    from ormatex_py.ormatex import PySysWrapped
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False
from ormatex_py.ode_sys import OdeSysNp

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False


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

    @jax.jit
    def _frhs(self, t, x, **kwargs):
        prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
        pred_t = self.delta * x[0] * x[1] - self.gamma * x[1]
        return jnp.array([prey_t, pred_t])

    # manually define the Jacobian LinOp (comment out to use autograd)
    @jax.jit
    def _fjac(self, t, x, **kwargs):
        jac = jnp.array([
            [self.alpha - self.beta * x[1], - self.beta * x[0]],
            [self.delta * x[1], self.delta * x[0] - self.gamma]
            ])
        fdt = jnp.zeros(x.shape) # supply zero fdt, to avoid finite difference
        return CustomJacLinOp(t, x, self.frhs, jac, fdt, frhs_kwargs=kwargs)

    # define a linear operator for testing
    @jax.jit
    def _fl(self, t, x, **kwargs):
        lop = jnp.array([
            [self.alpha - self.beta*2, 0.],
            [0., - self.gamma]
            ])
        return MatrixLinOp(lop)

@jax.jit
def f_pred_hunt(t):
    return 0.4 * (jnp.sin(t*0.2) + 1.0)

# manually define the time-derivative of f for testing
@jax.jit
def f_pred_hunt_dt(t):
    return 0.4 * 0.2 * jnp.cos(t*0.2)

class LotkaVolterraNonauto(OdeSplitSys):
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

    @jax.jit
    def _frhs(self, t, x, **kwargs):
        pred_hunt = f_pred_hunt(t)
        prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
        pred_t = self.delta * x[0] * x[1] - self.gamma * x[1] - pred_hunt * x[1]
        return jnp.array([prey_t, pred_t])

    # manually define the Jacobian LinOp (comment out to use autograd)
    @jax.jit
    def _fjac(self, t, x, **kwargs):
        pred_hunt = f_pred_hunt(t)
        jac = jnp.array([
            [self.alpha - self.beta * x[1], - self.beta * x[0]],
            [self.delta * x[1], self.delta * x[0] - self.gamma - pred_hunt]
            ])
        pred_hunt_dt = f_pred_hunt_dt(t)
        fdt = jnp.array([0, - pred_hunt_dt * x[1]])
        return CustomJacLinOp(t, x, self.frhs, jac, fdt, frhs_kwargs=kwargs)

    # define a linear operator for testing
    @jax.jit
    def _fl(self, t, x, **kwargs):
        lop = jnp.array([
            [self.alpha - self.beta*2, 0.],
            [0., - self.gamma]
            ])
        return MatrixLinOp(lop)


def main(method='epi3', do_plot=True, autonomous=True):
    # setup lotka voltera system
    t_end = 20.0
    y0 = jnp.array([0.1, 0.2])

    def gen_sys(autonomous=True):
        if not autonomous:
            return LotkaVolterraNonauto()
        else:
            return LotkaVolterra()

    # compute gold solution
    t0 = 0.0
    dt = 0.001
    nsteps = int(t_end/dt)
    tf = dt * nsteps
    step_ctrl = diffrax.ConstantStepSize()
    solver = diffrax.Dopri5()
    diffrax_lv_sys = diffrax.ODETerm(gen_sys(autonomous))
    res = diffrax.diffeqsolve(
            diffrax_lv_sys,
            solver,
            t0, tf, dt, y0,
            saveat=diffrax.SaveAt(steps=True),
            stepsize_controller=step_ctrl,
            max_steps=nsteps,
            )
    t_gold, y_gold = res.ts, res.ys

    if "_rs" in method:
        y0 = np.asarray(y0).reshape((-1, 1))
        lv_sys = PySysWrapped(OdeSysNp(gen_sys(autonomous)))
    else:
        lv_sys = gen_sys(autonomous)

    # sweep over time step size and integrate
    dt_list = [0.01, 0.0125, 0.02, 0.025, 0.05, 0.125]
    #dt_list = [0.5*dt for dt in dt_list]
    t_list = []
    y_list = []
    for dt in dt_list:
        print("dt = %0.2e" % dt)
        t0 = 0.0
        nsteps = int(t_end/dt)

        res = integrate_wrapper.integrate(
                lv_sys, y0, t0, dt, nsteps, method=method, max_krylov_dim=20, iom=20, tol_fdt=0)
        t_res, y_res = res.t_res, res.y_res
        t_res = jnp.asarray(t_res)
        y_res = jnp.asarray(y_res)
        t_list.append(t_res)
        y_list.append(y_res)

        if do_plot:
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
            plt.title(r"Method: %s, $\Delta t=$ %0.4f" % (method, dt))
            plt.tight_layout()
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
    lb = r"%s, O($\Delta t$)=%0.2e" % (method, popt[0])
    if do_plot:
        plt.figure()
        plt.scatter(err_dt[:, 0], err_dt[:, 1])
        plt.plot(err_dt[:, 0], np.exp(trendf(err_dt[:, 0], popt[0], popt[1])), ls='--', label=lb)
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r"$\Delta t$")
        plt.ylabel(r"|diff|")
        plt.grid(ls='--')
        plt.title("method: %s" % str(method))
        plt.tight_layout()
        plt.savefig("lotka_volterra_%s_conv.png" % (str(method)))
        plt.close()
    # time step sizes, error, label
    return err_dt, np.exp(trendf(err_dt[:, 0], popt[0], popt[1])), lb


def sweep_methods(autonomous=False):
    methods = ["epi3", "exprb2", "exprb3", "pexprb4", "implicit_euler", "implicit_esdirk3"]
    plt.figure()
    for method in methods:
        err_dt, err, lb = main(method, do_plot=False, autonomous=autonomous)
        plt.scatter(err_dt[:, 0], err_dt[:, 1])
        plt.plot(err_dt[:, 0], err, ls='--', label=lb)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"Time step size, $\Delta t$")
    plt.ylabel(r"|diff|")
    plt.grid(ls='--')
    plt.tight_layout()
    plt.savefig("lotka_volterra_sweep_conv.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    import diffrax
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default="epi3")
    parser.add_argument("-sweep", help="run convergence sweep", action="store_true", default=False)
    parser.add_argument("-nonautonomous", help="run nonautonomous system with external forcing", action="store_true", default=False)
    args = parser.parse_args()
    method = args.method

    # optionally run sweep over methods
    if args.sweep:
        sweep_methods(autonomous=(not args.nonautonomous))
    else:
        main(method, autonomous=(not args.nonautonomous))
