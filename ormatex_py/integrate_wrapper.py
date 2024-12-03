import numpy as np

import time

import jax
import jax.numpy as jnp

from ormatex_py.ode_sys import OdeSys
from ormatex_py.ode_exp import ExpRBIntegrator, ExpSplitIntegrator


def integrate(ode_sys, y0, t0, dt, nsteps, method, **kwargs):

    tic = time.perf_counter()

    is_rb = method in ExpRBIntegrator._valid_methods.keys()
    is_split = method in ExpSplitIntegrator._valid_methods.keys()
    if is_rb or is_split:
        # init the time integrator
        if is_rb:
            sys_int = ExpRBIntegrator(ode_sys, t0, y0, method=method, **kwargs)
        elif is_split:
            sys_int = ExpSplitIntegrator(ode_sys, t0, y0, method=method, **kwargs)

        t_res, y_res = integrate_ormatex(sys_int, y0, t0, dt, nsteps, method=method,
                                         **kwargs)
    else:
        try:
            # try integrate the system with diffrax
            t_res, y_res = integrate_diffrax(ode_sys, y0, t0, dt, nsteps, method=method)
        except AttributeError as e:
            print(e)
            raise AttributeError(f"no valid method {method} found")

    #wait for computation of last step to finish
    y_res[-1].block_until_ready()
    toc = time.perf_counter()

    print(f"Integrated system with {method} in {toc - tic:0.4f} seconds")
    return t_res, y_res


def integrate_diffrax(ode_sys, y0, t0, dt, nsteps, method="implicit_euler"):
    """
    Uses diffrax integrators to step adv diff system forward
    """
    import diffrax
    import optimistix
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
        root_finder=diffrax.VeryChord(rtol=1e-8, atol=1e-8, norm=optimistix.max_norm)
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
    return jnp.hstack(([t0], res.ts)), jnp.vstack((y0, res.ys))


def integrate_ormatex(sys_int, y0, t0, dt, nsteps, method="exprb2", **kwargs):
    """
    Uses ormatex exponential integrators to step adv diff system forward
    """
    t_res, y_res = [t0,], [y0,]
    for i in range(nsteps):
        res = sys_int.step(dt)
        # log the results for plotting
        t_res.append(res.t)
        y_res.append(res.y)
        # this would be where you could reject a step, if the
        # estimated err was too large
        sys_int.accept_step(res)
    return t_res, y_res
