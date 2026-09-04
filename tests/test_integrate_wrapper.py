import jax
import jax.numpy as jnp
import numpy as np

from ormatex_py.integrate_wrapper import TimeStepController, integrate_ormatex, integrate
from ormatex_py.ode_sys import StepResult, OdeSys


def callback_before_step_dummy(_sys: OdeSys, t: float, state: jax.Array):
    dummy_calc = jnp.sum(state) * t
    return {"dummy_array": jnp.asarray([1.0, 2.0,]), "dummy_calc": dummy_calc, "dummy_scale": 1.0}


class MockOdeSysWithCallback(OdeSys):
    """
    Simple mock ODE for testing that utilizes results from a callback function
    in the RHS.

    .. math::

        \frac{du}{dt} = c u
    """
    alpha: float

    def __init__(self, a=1.0):
        self.alpha = a
        super().__init__()

    @jax.jit
    def _frhs(self, t, x, **kwargs):
        dummy_array_0 = kwargs["dummy_array"][0]
        dummy_scale = kwargs["dummy_scale"]
        return (self.alpha * dummy_scale) * x * dummy_array_0


class MockIntegrator:
    order = 1
    sys = object()

    def __init__(self, errors):
        self.t = 0.0
        self.errors = iter(errors)
        self.calls = []

    def step(self, dt, frhs_kwargs=None):
        self.calls.append(dt)
        return StepResult(self.t + dt, dt, jnp.array([0.0]), next(self.errors))

    def accept_step(self, result):
        self.t = result.t


def _integrate_wrapper_with_callback(method="exprb3", phi_method="pfd"):
    sys = MockOdeSysWithCallback()
    dt = 2.0
    nsteps = 2
    t0, y0 = 0.0, jnp.array([1.0])
    res = integrate(sys, y0, t0, dt, nsteps, method=method, phi_method=phi_method,
                    callback_before_step=callback_before_step_dummy)
    t, y = np.asarray(res.t), np.asarray(res.y)
    assert np.isclose(t[-1], 4.0)
    assert np.isclose(y[-1], np.exp(4.0))
    # check callback results
    cb = res.cb
    assert np.isclose(cb["callback_before_step"][-1]["dummy_scale"], 1.0)
    callback_calc = cb["callback_before_step"][-1]["dummy_calc"]
    # should be dummy_calc with t=2.0, y=2.0  (start of last time step)
    assert np.isclose(callback_calc, 2.0*np.exp(2.0))


def test_integrate_wrapper_with_callback():
    _integrate_wrapper_with_callback("exprb2")
    _integrate_wrapper_with_callback("exprb3")
    _integrate_wrapper_with_callback("exprb4")
    _integrate_wrapper_with_callback("exprb2", phi_method="krylov")
    _integrate_wrapper_with_callback("exprb3", phi_method="krylov")
    _integrate_wrapper_with_callback("epi3", phi_method="krylov")
    _integrate_wrapper_with_callback("epi3", phi_method="leja")


def test_adaptive_steps_retry_and_reach_final_time():
    integrator = MockIntegrator([1.0, 0.0, 0.0])
    rejected = []
    controller = TimeStepController(atol=0.1, rtol=0.0)

    t_res, _, callback_res = integrate_ormatex(
        integrator,
        jnp.array([0.0]),
        0.0,
        0.1,
        1,
        step_controller=controller,
        callback_after_step_reject=lambda *args: rejected.append(args),
    )

    assert len(integrator.calls) == 3
    assert integrator.calls[1] < integrator.calls[0]
    assert t_res[-1] == 0.1
    assert len(rejected) == 1
    assert len(callback_res["callback_after_step_reject"]) == 1


def test_steps_without_error_estimate_are_always_accepted():
    integrator = MockIntegrator([-1.0, -1.0])

    class FailingController:
        def __call__(self, *args):
            raise AssertionError("controller should not handle missing estimates")

    t_res, _, _ = integrate_ormatex(
        integrator,
        jnp.array([0.0]),
        0.0,
        0.1,
        2,
        step_controller=FailingController(),
    )

    assert len(t_res) == 3
    assert t_res[-1] == 0.2


def test_no_controller_keeps_fixed_step_behavior():
    integrator = MockIntegrator([-1.0, -1.0])

    t_res, _, _ = integrate_ormatex(
        integrator, jnp.array([0.0]), 0.0, 0.1, 2
    )

    assert integrator.calls == [0.1, 0.1]
    assert t_res == [0.0, 0.1, 0.2]
