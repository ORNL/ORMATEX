import numpy as np
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp

from ormatex_py.ode_sys import OdeSys
from ormatex_py.ode_exp import ExpRBIntegrator, ExpSplitIntegrator, ExpLejaIntegrator
from ormatex_py.ode_explicit import RKIntegrator

try:
    from ormatex_py.ormatex import integrate_wrapper_rs, PySysWrapped
    from ormatex_py.ode_sys import OdeSysNp
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False


@dataclass
class IntegrateResult:
    """
    Storage for ORMATEX integration results
    """
    t: list
    y: list
    callback_res: dict
    err_code: int

    @property
    def t_res(self):
        return self.t

    @property
    def y_res(self):
        return self.y

    @property
    def cb(self):
        return self.callback_res


def integrate(ode_sys, y0, t0, dt, nsteps, method, **kwargs):
    """
    High level interface to all time integration methods in ORMATEX.
    """
    tic = time.perf_counter()
    is_rs = method in ["exprb2_rs", "exprb3_rs", "epi2_rs", "epi3_rs",
                       "bdf2_rs", "bdf1_rs", "backeuler_rs", "cn_rs"]
    is_rb = method in ExpRBIntegrator._valid_methods.keys()
    is_leja = method in ExpLejaIntegrator._valid_methods.keys()
    is_split = method in ExpSplitIntegrator._valid_methods.keys()
    is_rk = method in RKIntegrator._valid_methods.keys()
    c_res = {}
    if is_rb or is_split or is_rk or is_leja:
        # init the time integrator
        if is_rb:
            sys_int = ExpRBIntegrator(ode_sys, t0, y0, method=method, **kwargs)
        elif is_leja:
            sys_int = ExpLejaIntegrator(ode_sys, t0, y0, method=method, **kwargs)
        elif is_split:
            sys_int = ExpSplitIntegrator(ode_sys, t0, y0, method=method, **kwargs)
        elif is_rk:
            sys_int = RKIntegrator(ode_sys, t0, y0, method=method, **kwargs)

        t_res, y_res, c_res = integrate_ormatex(sys_int, y0, t0, dt, nsteps, method=method,
                                         **kwargs)
        #wait for computation of last step to finish
        y_res[-1].block_until_ready()
    elif is_rs:
        # try to integrate with rust ormatex integrators
        if not HAS_ORMATEX_RUST:
            raise ImportError("import ormatex_py.ormatex failed. Rust ormatex bindings not found. Run: maturin develop --release")
        if not isinstance(ode_sys, PySysWrapped):
            ode_sys = PySysWrapped(OdeSysNp(ode_sys))
        y_res, t_res = integrate_wrapper_rs(ode_sys, np.asarray(y0).reshape((-1, 1)), t0, dt, nsteps, method=str(method[0:-3]), **kwargs)
        y_res, t_res = np.asarray(y_res).squeeze(), np.asarray(t_res)
    else:
        try:
            # try integrate the system with diffrax
            t_res, y_res = integrate_diffrax(ode_sys, y0, t0, dt, nsteps, method=method, **kwargs)
        except AttributeError as e:
            print(e)
            raise AttributeError(f"no valid method {method} found")

        #wait for computation of last step to finish
        y_res[-1].block_until_ready()
    toc = time.perf_counter()

    print(f"Integrated system with {method} in {toc - tic:0.4f} seconds")
    return IntegrateResult(t_res, y_res, c_res, 0)


def integrate_diffrax(ode_sys, y0, t0, dt, nsteps, method="implicit_euler", **kwargs):
    """
    Uses diffrax integrators to step adv diff system forward
    """
    import diffrax
    import optimistix
    import lineax
    # thin wrapper around ode_sys for diffrax compat
    diffrax_ode_sys = diffrax.ODETerm(ode_sys)
    method_dict = {
            # explicit
            "euler": diffrax.Euler,
            "heun": diffrax.Heun,
            "midpoint": diffrax.Midpoint,
            "bosh3": diffrax.Bosh3,
            "dopri5": diffrax.Dopri5,
            # implicit
            "implicit_euler": diffrax.ImplicitEuler,
            "implicit_esdirk3": diffrax.Kvaerno3,
            "implicit_esdirk4": diffrax.Kvaerno4,
           }

    if not method in method_dict.keys():
        raise AttributeError(f"{method} not in diffrax")

    try:
        root_finder=optimistix.Newton(
                rtol=kwargs.get("rtol", 1e-8),
                atol=kwargs.get("atol", 1e-8),
                linear_solver=lineax.GMRES(
                    rtol=kwargs.get("lin_rtol", 1e-8),
                    atol=kwargs.get("lin_atol", 1e-8),
                    restart=2000, stagnation_iters=2000),
                )
        solver = method_dict[method](root_finder=root_finder)
    except:
        solver = method_dict[method]()
    tf = dt * nsteps
    step_ctrl = diffrax.ConstantStepSize()
    res = diffrax.diffeqsolve(
            diffrax_ode_sys,
            solver,
            t0, tf, dt, y0,
            saveat=diffrax.SaveAt(steps=True),
            stepsize_controller=step_ctrl,
            max_steps=nsteps,
            )
    # return res.ts, res.ys
    return jnp.hstack((jnp.asarray([t0]), res.ts)), jnp.vstack((y0, res.ys))


def integrate_ormatex(sys_int, y0, t0, dt, nsteps, method="exprb2", **kwargs):
    """
    Uses ormatex exponential integrators to step adv diff system forward
    """
    t_res, y_res = [t0,], [y0,]
    callback_before_step = kwargs.get("callback_before_step", None)
    callback_after_step_accept = kwargs.get("callback_after_step_accept", None)
    callback_after_step_reject = kwargs.get("callback_after_step_reject", None)
    callback_res = {"callback_before_step": [], "callback_after_step_accept": []}
    for i in range(nsteps):
        if callable(callback_before_step):
            callback_res["callback_before_step"].append(
                    callback_before_step(sys_int.sys, t_res[-1], y_res[-1]))
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)
        # callbacks
        if callable(callback_after_step_accept):
            callback_res["callback_after_step_accept"].append(
                    callback_after_step_accept(sys_int.sys, t_res[-1], y_res[-1]))
    return t_res, y_res, callback_res
