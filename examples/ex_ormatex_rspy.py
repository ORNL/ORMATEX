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
        pred_hunt = 0.4*(jnp.sin(t*0.2)+1.0)
        # pred prey time derivatives
        prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
        pred_t = self.delta * x[0] * x[1] - self.gamma * x[1] - pred_hunt*x[1]
        res = jnp.asarray([prey_t, pred_t])
        return jax.device_get(res).flatten()

def run_model():
    # Wrap the system for rust compatibility
    lv_sys = PySysWrapped(OdeSysNp(LotkaVolterra()))
    # Step the system forward using rust-based integrator
    t0 = 0.0
    y0 = np.array([[0.1, 0.2],]).T
    dt = 0.05
    nsteps = 1000
    res = integrate(lv_sys, y0, t0, dt, nsteps, method="exprb2_rs", tol_fdt=1.0e-6)
    # y0 = jnp.array([0.1, 0.2])
    # res = integrate(LotkaVolterra(), y0, t0, dt, nsteps, method="implicit_esdirk3", tol_fdt=1.0e-6)
    return res.t_res, res.y_res

if __name__ == "__main__":
    import matplotlib.pylab as plt
    t_out, y_out = run_model()
    # Visualize results
    print(y_out)
    plt.figure()
    plt.plot(t_out, y_out[:, 0], label='prey')
    plt.plot(t_out, y_out[:, 1], label='pred')
    plt.plot(t_out, 0.4*(np.sin(t_out*0.2)+1.0), label='predator hunters')
    plt.grid(ls='--')
    plt.legend()
    plt.savefig("ormatex_rspy_lv.png")
    plt.close()
