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
    ft_scale: float

    def __init__(self, a=1.0, b=1.0, d=1.0, g=1.0, **kwargs):
        self.alpha = a
        self.beta = b
        self.delta = d
        self.gamma = g
        self.ft_scale = kwargs.get("ft_scale", 1.0)
        super().__init__()

    @jax.jit
    def _frhs(self, t, x, **kwargs):
        # hunter populations are known functions of time
        pred_hunt = f_pred_hunt(t) * self.ft_scale
        # pred prey time derivatives
        prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
        pred_t = self.delta * x[0] * x[1] - self.gamma * x[1] - pred_hunt*x[1]
        res = jnp.asarray([prey_t, pred_t])
        return jax.device_get(res).flatten()

def run_model(dt, nsteps, method="exprb2_rs", tol_fdt=1.0e-6, ft_scale=1.0):
    # Step the system forward
    t0 = 0.0
    y0 = np.array([0.1, 0.2])
    res = integrate(LotkaVolterra(ft_scale=ft_scale), y0, t0, dt, nsteps,
                    method=method, tol_fdt=tol_fdt)
    y0 = jnp.array(y0.flatten())
    # Check against dopri5 in diffrax
    res_expected = integrate(LotkaVolterra(ft_scale=ft_scale), y0, t0, dt, nsteps,
                             method="dopri5")
    return np.asarray(res.t), np.asarray(res.y), res_expected.t, res_expected.y

if __name__ == "__main__":
    import matplotlib.pylab as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", help="Integration method. Valid methods are: "
                        "exprb2_rs, exprb3_rs, epi3_rs, exprb2, exprb3, epi3. "
                        "Methods ending in _rs are rust impl. Others are python/JAX impl.",
                        type=str, default="epi3_rs")
    parser.add_argument("-ft_scale", help="Forcing term scale", type=float, default=1.0)
    parser.add_argument("-dt", help="time step size", type=float, default=0.05)
    parser.add_argument("-nsteps", help="number of steps", type=int, default=1000)
    parser.add_argument("-tol_fdt", help="Nonautonomous system check threshold", type=float, default=1.0e-6)
    args = parser.parse_args()
    t_out, y_out, t_true, y_true = run_model(
            args.dt, args.nsteps, args.method, args.tol_fdt, ft_scale=args.ft_scale)
    # Visualize results
    print(y_out)
    plt.figure()
    plt.plot(t_out, y_out[:, 0], label='prey')
    plt.plot(t_out, y_out[:, 1], label='pred')
    plt.plot(t_true, y_true[:, 0], ls='--', label='prey true')
    plt.plot(t_true, y_true[:, 1], ls='--', label='pred true')
    mae = np.mean(np.abs(y_out[:, 0] - y_true[:, 0]))
    plt.plot(t_out, f_pred_hunt(t_out)*args.ft_scale, label='predator hunters')
    plt.grid(ls='--')
    plt.title("Method: %s, MAE err: %0.3e" % (args.method, mae))
    plt.legend()
    plt.savefig("ormatex_rspy_lv_%s.png" % (str(args.method)))
    plt.close()
