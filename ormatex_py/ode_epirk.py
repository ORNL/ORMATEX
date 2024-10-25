"""
EPIRK time integration methods
"""
import jax
import jax.numpy as jnp
from typing import Callable
from ormatex_py.ode_sys import IntegrateSys, OdeSys, StepResult
from ormatex_py.matexp_krylov import phi_linop, matexp_linop


class EpirkIntegrator(IntegrateSys):
    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, method="epirk2", *args, **kwargs):
        valid_methods = {"epirk2": 2, "epirk3": 3}
        self.method = method
        assert self.method in list(valid_methods.keys())
        order = valid_methods[self.method]
        super().__init__(sys, t0, y0, order, method, *args, **kwargs)
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        self.iom = kwargs.get("iom", 2)

    def _remf(self, tr: float, yr: jax.Array,
              frhs_y0: jax.Array, sys_jac_lop_y0: Callable):
        """
        Computes remainder R(yr) = frhs(yr) - frhs(y0) - J_y0*(yr-y0)
        """
        y0 = self.y_hist[0]
        frhs_yr = self.sys.frhs(tr, yr)
        jac_yd = sys_jac_lop_y0(yr - y0)
        return frhs_yr - frhs_y0 - jac_yd

    def _step_epirk2(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        u_{t+1} = u_t + dt*\varphi_1(dt*J_t)F(t, u_t)
        """
        t = self.t
        y0 = self.y_hist[0]
        sys_jac_lop = self.sys.fjac(t, y0)
        fy0 = self.sys.frhs(t, y0)
        fy0_dt = fy0 * dt
        y_new = y0 + phi_linop(
                sys_jac_lop, dt, fy0_dt, 1, self.max_krylov_dim, self.iom)
        # no error est. avail
        y_err = -1.
        return StepResult(t+dt, dt, y_new, y_err)

    def _step_epirk3(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        u_{t+1} = u_t + dt*\varphi_1(dt*J_t)F(t, u_t) +
            (2/3)*dt*\varphi_2(dt*J_t)R(t, u_t, u_{t-1})
        """
        t = self.t
        y0 = self.y_hist[0] # y_t
        yp = self.y_hist[1] # y_{t-1}
        tp = self.t_hist[1]

        sys_jac_lop = self.sys.fjac(t, y0)
        fy0 = self.sys.frhs(t, y0)
        fy0_dt = fy0 * dt
        y1 = y0 + phi_linop(
                sys_jac_lop, dt, fy0_dt, 1, self.max_krylov_dim, self.iom)

        rn_dt = self._remf(tp, yp, fy0, sys_jac_lop) * dt
        y_new = y1 + (2./3.)*phi_linop(
                sys_jac_lop, dt, rn_dt, 2, self.max_krylov_dim, self.iom)
        y_err = jnp.max(jnp.abs(y1 - y_new))
        return StepResult(t+dt, dt, y_new, y_err)

    def step(self, dt: float) -> StepResult:
        if self.method == "epirk2":
            return self._step_epirk2(dt)
        elif self.method == "epirk3":
            if len(self.y_hist) >= 2:
                return self._step_epirk3(dt)
            else:
                return self._step_epirk2(dt)
        else:
            raise NotImplementedError

    def accept_step(self, s: StepResult):
        self.t = s.t
        self.t_hist.appendleft(s.t)
        self.y_hist.appendleft(s.y)
