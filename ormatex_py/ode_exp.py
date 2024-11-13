"""
Exponential time integration methods
"""
import jax
import jax.numpy as jnp
from ormatex_py.ode_sys import LinOp, IntegrateSys, OdeSys, OdeSysSplit, StepResult
from ormatex_py.matexp_krylov import phi_linop, matexp_linop, kiops_fixedsteps

##TODO:
# - RB methods are only second and third order for f(t,y) = f(y) non-autonomous systems
#   time dependent problems require a correction involving f'(t,y)
#   see: https://doi.org/10.1137/080717717
class ExpRBIntegrator(IntegrateSys):
    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, method="epi2", *args, **kwargs):
        valid_methods = {"exprb2": 2, "exprb3": 3, "epi2": 2, "epi3": 3}
        self.method = method
        assert self.method in list(valid_methods.keys())
        order = valid_methods[self.method]
        super().__init__(sys, t0, y0, order, method, *args, **kwargs)
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        self.iom = kwargs.get("iom", 2)

    def _remf(self, tr: float, yr: jax.Array,
              frhs_y0: jax.Array, sys_jac_lop_y0: LinOp):
        """
        Computes remainder R(yr) = frhs(yr) - frhs(y0) - J_y0*(yr-y0)
        """
        y0 = self.y_hist[0]
        frhs_yr = self.sys.frhs(tr, yr)
        jac_yd = sys_jac_lop_y0(yr - y0)
        return frhs_yr - frhs_y0 - jac_yd

    def _step_exprb2(self, dt: float) -> StepResult:
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

    # lowest order EPI multistep method is single step
    _step_epi2 = _step_exprb2

    def _step_epi3(self, dt: float) -> StepResult:
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
        rn_dt = self._remf(tp, yp, fy0, sys_jac_lop) * dt

        # use kiops to save 1 call to arnoldi. almost 2x speedup
        vb0 = jnp.zeros(y0.shape)
        y_update = kiops_fixedsteps(
            sys_jac_lop, dt, [vb0, fy0_dt, (2./3.)*rn_dt],
            max_krylov_dim=self.max_krylov_dim, iom=self.iom)
        y_new = y0 + y_update
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)

    def _step_exprb3(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        u_{t+1} = u_t + dt*\varphi_1(dt*J_t)F(t, u_t) +
            2*dt*\varphi_3(dt*J_t)R_2
        """
        t = self.t
        y0 = self.y_hist[0] # y_t

        sys_jac_lop = self.sys.fjac(t, y0)
        fy0 = self.sys.frhs(t, y0)

        # 2nd stage
        t_2 = t + dt
        y_2 = y0 + dt*phi_linop(sys_jac_lop, dt, fy0, 1,
                                self.max_krylov_dim, self.iom)
        r_2 = self._remf(t_2, y_2, fy0, sys_jac_lop)

        # compute final update
        y_new = y_2 + 2.*dt*phi_linop(sys_jac_lop, dt, r_2, 3,
                                      self.max_krylov_dim, self.iom)

        # TODO: error estimate by comparing y_2 and y_new?
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)


    def step(self, dt: float) -> StepResult:
        if self.method == "exprb2":
            return self._step_exprb2(dt)
        elif self.method == "exprb3":
            return self._step_exprb3(dt)
        elif self.method == "epi2":
            return self._step_epi2(dt)
        elif self.method == "epi3":
            if len(self.y_hist) >= 2:
                return self._step_epi3(dt)
            else:
                return self._step_epi2(dt)
        else:
            raise NotImplementedError

    def accept_step(self, s: StepResult):
        self.t = s.t
        self.t_hist.appendleft(s.t)
        self.y_hist.appendleft(s.y)


class ExpSplitIntegrator(IntegrateSys):
    def __init__(self, sys: OdeSysSplit, t0: float, y0: jax.Array, method="exp3", *args, **kwargs):
        valid_methods = {"exp1": 1, "exp2": 2, "exp3": 3}
        self.method = method
        assert self.method in list(valid_methods.keys())
        order = valid_methods[self.method]
        super().__init__(sys, t0, y0, order, method, *args, **kwargs)
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        self.iom = kwargs.get("iom", 2)

    def _remf(self, tr: float, yr: jax.Array,
              frhs_y0: jax.Array, sys_lop_y0: LinOp):
        """
        Computes remainder R(yr) = frhs(yr) - frhs(y0) - L_y0*(yr-y0)
        """
        y0 = self.y_hist[0]
        frhs_yr = self.sys.frhs(tr, yr)
        L_yd = sys_lop_y0(yr - y0)
        return frhs_yr - frhs_y0 - L_yd

    def _step_exp1(self, dt: float) -> StepResult:
        """
        Exponential Euler,
        computes the solution update by:
        u_{t+1} = u_t + dt*\varphi_1(dt*L_t)F(t, u_t)
        """
        t = self.t
        y0 = self.y_hist[0]
        sys_lop = self.sys.fl(t, y0)
        fy0 = self.sys.frhs(t, y0)
        fy0_dt = fy0 * dt
        y_new = y0 + phi_linop(
                sys_lop, dt, fy0_dt, 1, self.max_krylov_dim, self.iom)
        # no error est. avail
        y_err = -1.
        return StepResult(t+dt, dt, y_new, y_err)

    def _step_exp2(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        u_2 = u_t + c2*dt*\varphi_1(c2*dt*L_t)F(t, u_t)
        u_{t+1} = u_t + dt*\varphi_1(dt*L_t)F(t, u_t)
                      + dt/c2*\varphi_2(dt*L_t)R_2
        """
        t = self.t
        y0 = self.y_hist[0] # y_t

        sys_lop = self.sys.fl(t, y0)
        fy0 = self.sys.frhs(t, y0)

        c2 = 1. #constant between 0 and 1

        # 2nd stage
        t_2 = t + c2*dt
        y_2 = y0 + phi_linop(sys_lop, c2*dt, c2*dt*fy0, 1,
                             self.max_krylov_dim, self.iom)
        r_2 = self._remf(t_2, y_2, fy0, sys_lop)

        # compute final update
        ## TODO: only valid for c2=1., otherwise reuse Krylov, or use KIOPS
        y_new = y_2 + phi_linop(sys_lop, dt, r_2 * dt/c2, 2,
                                self.max_krylov_dim, self.iom)

        # TODO: error estimate by comparing y_2 and y_new?
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)

    def _step_exp3(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        u_2 = u_t + c2*dt*\varphi_1(c2*dt*L_t)F(t, u_t)
        u_3 = u_t + (2/3)*dt*\varphi_1((2/3)*dt*L_t)F(t, u_t) + (4./9./c2)*dt*\varphi_2((2/3)*dt*L_t)R_2
        u_{t+1} = u_t + dt*\varphi_1(dt*L_t)F(t, u_t) + 3/2*dt*\varphi_2(dt*L_t)R_3
        """
        t = self.t
        y0 = self.y_hist[0] # y_t

        sys_lop = self.sys.fl(t, y0)
        fy0 = self.sys.frhs(t, y0)

        c2 = 2./3. #constant between 0 and 1

        # 2nd stage
        t_2 = t + c2*dt
        y_2 = y0 + phi_linop(sys_lop, dt*c2, fy0*dt*c2, 1,
                             self.max_krylov_dim, self.iom)
        r_2 = self._remf(t_2, y_2, fy0, sys_lop)

        # 3rd stage
        ## TODO: only valid for c2=2/3, otherwise reuse Krylov, or use KIOPS
        t_3 = t + (2./3.)*dt
        y_3 = y_2 + phi_linop(sys_lop, dt*(2./3.), r_2 * dt*(4./9./c2), 2,
                              self.max_krylov_dim, self.iom)
        r_3 = self._remf(t_3, y_3, fy0, sys_lop)

        # compute final update
        y_new = y0 \
              + phi_linop(sys_lop, dt, fy0*dt, 1, self.max_krylov_dim, self.iom) \
              + phi_linop(sys_lop, dt, r_3*dt*(3./2.), 2, self.max_krylov_dim, self.iom)

        # TODO: embedded error estimate?
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)


    def step(self, dt: float) -> StepResult:
        if self.method == "exprb2":
            return self._step_exprb2(dt)
        elif self.method == "exprb3":
            return self._step_exprb3(dt)
        elif self.method == "epi2":
            return self._step_epi2(dt)
        elif self.method == "epi3":
            if len(self.y_hist) >= 2:
                return self._step_epi3(dt)
            else:
                return self._step_epi2(dt)
        else:
            raise NotImplementedError

    def accept_step(self, s: StepResult):
        self.t = s.t
        self.t_hist.appendleft(s.t)
        self.y_hist.appendleft(s.y)
