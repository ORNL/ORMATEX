# Example using python bindings to ormatex rust integrators
#
# NOTE: Before executing this example demo.  Ensure the ormatex
# package is installed by running the following:
#
#     maturin develop --release
#
import numpy as np
import jax
from jax import numpy as jnp
from ormatex_py.integrate_wrapper import integrate
from ormatex_py.ormatex import PySysWrapped, integrate_wrapper_rs
from ormatex_py.ode_sys import OdeSys, OdeSysNp, MatrixLinOp
jax.config.update("jax_enable_x64", True)


@jax.jit
def f_pred_hunt(t):
    return 0.4*(jnp.sin(t*0.2)+1.0)

# Define the nonautonomous ODE system
class LotkaVolterra(OdeSys):
    alpha: float
    beta: float
    delta: float
    gamma: float

    def __init__(self, a=1.0, b=1.0, d=1.0, g=1.0, **kwargs):
        self.alpha = a
        self.beta = b
        self.delta = d
        self.gamma = g
        super().__init__()

    @jax.jit
    def _frhs(self, t, x, **kwargs):
        # hunter populations are known functions of time
        pred_hunt = f_pred_hunt(t)
        # pred prey time derivatives
        prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
        pred_t = self.delta * x[0] * x[1] - self.gamma * x[1] - pred_hunt*x[1]
        res = jnp.asarray([prey_t, pred_t])
        return jax.device_get(res).flatten()

def run_model(method="exprb2_rs", tol_fdt=1.0e-6):
    # Wrap the system for rust compatibility
    lv_sys = PySysWrapped(OdeSysNp(LotkaVolterra()))
    # Step the system forward using rust-based integrator
    t0 = 0.0
    y0 = np.array([[0.1, 0.2],]).T
    dt = 0.05
    nsteps = 1000
    if "_rs" in method:
        res = integrate(lv_sys, y0, t0, dt, nsteps, method=method, tol_fdt=tol_fdt)
    else:
        y0 = jnp.array(y0.flatten())
        res = integrate(LotkaVolterra(), y0, t0, dt, nsteps, method=method, tol_fdt=tol_fdt)
    y0 = jnp.array(y0.flatten())
    res_expected = integrate(LotkaVolterra(), y0, t0, dt, nsteps, method="dopri5")
    return res.t, res.y, res_expected.t, res_expected.y

if __name__ == "__main__":
    import matplotlib.pylab as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", help="Integration method", type=str, default="exprb2_rs")
    parser.add_argument("-tol_fdt", help="Nonautonomous system check threshold", type=float, default=1.0e-6)
    args = parser.parse_args()
    t_out, y_out, t_true, y_true = run_model(args.method, args.tol_fdt)
    # Visualize results
    print(y_out)
    plt.figure()
    plt.plot(t_out, y_out[:, 0], label='prey')
    plt.plot(t_out, y_out[:, 1], label='pred')
    plt.plot(t_true, y_true[:, 0], ls='--', label='prey true')
    plt.plot(t_true, y_true[:, 1], ls='--', label='pred true')
    mae = np.mean(np.abs(y_out[:, 0] - y_true[:, 0]))
    plt.plot(t_out, f_pred_hunt(t_out), label='predator hunters')
    plt.grid(ls='--')
    plt.title("Method: %s, MAE err: %0.3e" % (args.method, mae))
    plt.legend()
    plt.savefig("ormatex_rspy_lv_%s.png" % (str(args.method)))
    plt.close()
