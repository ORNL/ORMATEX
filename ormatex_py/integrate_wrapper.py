import numpy as np

import time

import jax
import jax.numpy as jnp

from ormatex_py.ode_sys import OdeSys
from ormatex_py.ode_exp import ExpRBIntegrator, ExpSplitIntegrator
from ormatex_py.ode_explicit import RKIntegrator

try:
    from ormatex_py.ormatex import integrate_wrapper_rs, PySysWrapped
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False

def integrate(ode_sys, y0, t0, dt, nsteps, method, **kwargs):
    tic = time.perf_counter()
    is_rs = method in ["exprb2_rs", "exprb3_rs", "epi2_rs", "epi3_rs",
                       "bdf2_rs", "bdf1_rs", "backeuler_rs", "cn_rs"]
    is_rb = method in ExpRBIntegrator._valid_methods.keys()
    is_split = method in ExpSplitIntegrator._valid_methods.keys()
    is_rk = method in RKIntegrator._valid_methods.keys()
    c_res = None
    if is_rb or is_split or is_rk:
        # init the time integrator
        if is_rb:
            sys_int = ExpRBIntegrator(ode_sys, t0, y0, method=method, **kwargs)
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
            raise ImportError("import ormatex_py.ormatex failed. Rust ormatex bindings not found. Run: maturin develop")
        assert isinstance(ode_sys, PySysWrapped)
        y_res, t_res = integrate_wrapper_rs(ode_sys, y0, t0, dt, nsteps, method=str(method[0:-3]), **kwargs)
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
    if c_res is not None and "callback_fns" in kwargs.keys():
        return t_res, y_res, c_res
    else:
        return t_res, y_res


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
    callback_fn_dict = kwargs.get("callback_fns", {})
    callback_res = {key: [] for key in callback_fn_dict.keys()}
    for i in range(nsteps):
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)
        # callbacks
        for fn_name, cb_fn in callback_fn_dict.items():
            # compute result of callback
            cb_res = cb_fn(sys_int.sys, res.t, res.y)
            callback_res[fn_name].append(cb_res)
    return t_res, y_res, callback_res
