"""
Exponential time integration methods
"""
import jax
import jax.numpy as jnp

from functools import partial

from ormatex_py.ode_sys import LinOp, IntegrateSys, OdeSys, OdeSplitSys, StepResult
from ormatex_py.matexp_krylov import phi_linop, matexp_linop, kiops_fixedsteps
from ormatex_py.matexp_phi import f_phi_k_ext, f_phi_k_sq_all

##TODO:
# - RB methods are only second and third order for f(t,y) = f(y) non-autonomous systems
#   time dependent problems require a correction involving f'(t,y)
#   see: https://doi.org/10.1137/080717717
class ExpRBIntegrator(IntegrateSys):

    _valid_methods = {"exprb2": 2, "exprb3": 3, "epi2": 2, "epi3": 3,
                      "exprb2_dense": 2}

    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, method="epi2", **kwargs):
        self.method = method
        if not self.method in self._valid_methods.keys():
            raise AttributeError(f"{self.method} not in {self._valid_methods}")
        order = self._valid_methods[self.method]
        super().__init__(sys, t0, y0, order, method, **kwargs)
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        self.iom = kwargs.get("iom", 100)

    def reset_ic(self, t0: float, y0: jax.Array):
        super().reset_ic(t0, y0)

    def _remf(self, tr: float, yr: jax.Array,
              frhs_yt: jax.Array, sys_jac_lop_yt: LinOp):
        """
        Computes remainder R(yr) = frhs(yr) - frhs(yt) - J_yt*(yr-yt)
        """
        yt = self.y_hist[0]
        frhs_yr = self.sys.frhs(tr, yr)
        jac_yd = sys_jac_lop_yt(yr - yt)
        return frhs_yr - frhs_yt - jac_yd

    def _step_exprb2(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t)

        doi:
        """
        t = self.t
        yt = self.y_hist[0]
        sys_jac_lop = self.sys.fjac(t, yt)
        fyt = self.sys.frhs(t, yt)
        fyt_dt = fyt * dt
        y_new = yt + phi_linop(
                sys_jac_lop, dt, fyt_dt, 1, self.max_krylov_dim, self.iom)
        # no error est. avail
        y_err = -1.
        return StepResult(t+dt, dt, y_new, y_err)

    # lowest order EPI multistep method is single step
    _step_epi2 = _step_exprb2

    def _step_epi3(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t) +
            (2/3)*dt*\varphi_2(dt*J_t)R(t, y_t, y_{t-1})

        doi:
        """
        t = self.t
        yt = self.y_hist[0] # y_t
        yp = self.y_hist[1] # y_{t-1}
        tp = self.t_hist[1]

        sys_jac_lop = self.sys.fjac(t, yt)
        fyt = self.sys.frhs(t, yt)
        fyt_dt = fyt * dt
        rn_dt = self._remf(tp, yp, fyt, sys_jac_lop) * dt

        # use kiops to save 1 call to arnoldi. almost 2x speedup
        vb0 = jnp.zeros(yt.shape)
        y_update = kiops_fixedsteps(
            sys_jac_lop, dt, [vb0, fyt_dt, (2./3.)*rn_dt],
            max_krylov_dim=self.max_krylov_dim, iom=self.iom)
        y_new = yt + y_update
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)

    def _step_exprb3(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t) +
            2*dt*\varphi_3(dt*J_t)R_2

        doi:
        """
        t = self.t
        yt = self.y_hist[0] # y_t

        sys_jac_lop = self.sys.fjac(t, yt)
        fyt = self.sys.frhs(t, yt)

        # 2nd stage
        t_2 = t + dt
        y_2 = yt + dt*phi_linop(sys_jac_lop, dt, fyt, 1,
                                self.max_krylov_dim, self.iom)
        r_2 = self._remf(t_2, y_2, fyt, sys_jac_lop)

        # compute final update
        y_new = y_2 + 2.*dt*phi_linop(sys_jac_lop, dt, r_2, 3,
                                      self.max_krylov_dim, self.iom)

        # TODO: error estimate by comparing y_2 and y_new?
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)


    @jax.jit
    def _step_exprb2_jit(t, yt, dt, sys):
        print("jit-compiling exprb2_dense kernel")
        fyt = sys.frhs(t, yt)

        J = sys.fjac(t, yt).dense()
        phi1J = f_phi_k_sq_all(dt*J, 1)[1]
        #phi1J = f_phi_k_ext(dt*J, 1)
        y_new = yt + dt * phi1J @ fyt
        # no error est. avail
        y_err = -1.
        return y_new, y_err

    def _step_exprb2_dense(self, dt: float) -> StepResult:
        """
        Exponential Euler,
        computes the solution update by:
        y_{t+1} = y_t + dt*\varphi_1(dt*L)F(t, y_t)
        """
        t = self.t
        yt = self.y_hist[0]

        y_new, y_err = ExpRBIntegrator._step_exprb2_jit(t, yt, dt, self.sys)

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
        elif self.method == "exprb2_dense":
            return self._step_exprb2_dense(dt)
        else:
            raise NotImplementedError

    def accept_step(self, s: StepResult):
        self.t = s.t
        self.t_hist.appendleft(s.t)
        self.y_hist.appendleft(s.y)


class ExpSplitIntegrator(IntegrateSys):

    _valid_methods = {"exp1": 1, "exp2": 2, "exp3": 3,
                      "exp1_dense": 1, "exp2_dense": 2, "exp3_dense": 3}

    def __init__(self, sys: OdeSplitSys, t0: float, y0: jax.Array, method="exp3", **kwargs):
        self.method = method
        if not self.method in self._valid_methods.keys():
            raise AttributeError
        order = self._valid_methods[self.method]
        super().__init__(sys, t0, y0, order, method, **kwargs)

        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        self.iom = kwargs.get("iom", 200)

        #TODO: used to check if dense phiL matrices need updating
        self._cached_dt = None

    def reset_ic(self, t0: float, y0: jax.Array):
        super().reset_ic(t0, y0)
        self._cached_dt = None

    def _remf(self, tr: float, yr: jax.Array,
              frhs_yt: jax.Array, sys_lop: LinOp):
        """
        Computes remainder R(yr) = frhs(yr) - frhs(yt) - L*(yr-y0)
        """
        yt = self.y_hist[0]
        frhs_yr = self.sys.frhs(tr, yr)
        L_yd = sys_lop(yr - yt)
        return frhs_yr - frhs_yt - L_yd

    def _step_exp1(self, dt: float) -> StepResult:
        """
        Exponential Euler,
        computes the solution update by:
        y_{t+1} = y_t + dt*\varphi_1(dt*L_t)F(t, u_t)
        """
        t = self.t
        yt = self.y_hist[0]
        sys_lop = self.sys.fl(t, yt)
        fyt = self.sys.frhs(t, yt)
        fyt_dt = fyt * dt
        y_new = yt + phi_linop(
                sys_lop, dt, fyt_dt, 1, self.max_krylov_dim, self.iom)
        # no error est. avail
        y_err = -1.
        return StepResult(t+dt, dt, y_new, y_err)

    def _step_exp2(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        y_2 = y_t + c2*dt*\varphi_1(c2*dt*L_t)F(t, y_t)
        y_{t+1} = y_t + dt*\varphi_1(dt*L_t)F(t, y_t)
                      + dt/c2*\varphi_2(dt*L_t)R_2
        """
        t = self.t
        yt = self.y_hist[0]

        sys_lop = self.sys.fl(t, yt)
        fyt = self.sys.frhs(t, yt)

        c2 = 1. #constant between 0 and 1

        # 2nd stage
        t_2 = t + c2*dt
        y_2 = yt + phi_linop(sys_lop, c2*dt, c2*dt*fyt, 1,
                             self.max_krylov_dim, self.iom)
        r_2 = self._remf(t_2, y_2, fyt, sys_lop)

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
        y_2 = y_t + c2*dt*\varphi_1(c2*dt*L_t)F(t, y_t)
        y_3 = y_t + (2/3)*dt*\varphi_1((2/3)*dt*L_t)F(t, y_t) + (4./9./c2)*dt*\varphi_2((2/3)*dt*L_t)R_2
        y_{t+1} = y_t + dt*\varphi_1(dt*L_t)F(t, y_t) + 3/2*dt*\varphi_2(dt*L_t)R_3
        """
        t = self.t
        yt = self.y_hist[0]

        sys_lop = self.sys.fl(t, yt)
        fyt = self.sys.frhs(t, yt)

        c2 = 2./3. #constant between 0 and 1

        # 2nd stage
        t_2 = t + c2*dt
        y_2 = yt + phi_linop(sys_lop, dt*c2, fyt*dt*c2, 1,
                             self.max_krylov_dim, self.iom)
        r_2 = self._remf(t_2, y_2, fyt, sys_lop)

        # 3rd stage
        ## TODO: only valid for c2=2/3, otherwise reuse Krylov, or use KIOPS
        t_3 = t + (2./3.)*dt
        y_3 = y_2 + phi_linop(sys_lop, dt*(2./3.), r_2 * dt*(4./9./c2), 2,
                              self.max_krylov_dim, self.iom)
        r_3 = self._remf(t_3, y_3, fyt, sys_lop)

        # compute final update
        y_new = yt \
              + phi_linop(sys_lop, dt, fyt*dt, 1, self.max_krylov_dim, self.iom) \
              + phi_linop(sys_lop, dt, r_3*dt*(3./2.), 2, self.max_krylov_dim, self.iom)

        # TODO: embedded error estimate?
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)


    def _update_dense_phiLs(self, dt: float, cks: list[tuple[float,list[int]]]):
        if not (self._cached_dt and self._cached_dt == dt):
            # TODO: L is only evaluated the first step (t0, yt) or if dt changes
            #   for some examples, want to update L once t or y changes substantially
            self._cached_sys_lop = self.sys.fl(self.t, self.y_hist[0])
            L = self._cached_sys_lop.dense()

            self._cached_phiLs = dict()
            for c, ks in cks:
                # use scaling and modified squaring
                phikLs = f_phi_k_sq_all(dt*c*L, max(ks))
                #phikLs = f_phi_k_ext(dt*c*L, max(ks), return_all=True)
                for k in ks:
                    self._cached_phiLs[(c,k)] = phikLs[k]
            self._cached_dt = dt

        return self._cached_phiLs

    @jax.jit
    def _step_exp1_jit(t, yt, dt, sys, phiLs):
        print("jit-compiling exp1_dense kernel")
        fyt = sys.frhs(t, yt)
        y_new = yt + dt * phiLs[(1.,1)] @ fyt
        # no error est. avail
        y_err = -1.
        return y_new, y_err

    def _step_exp1_dense(self, dt: float) -> StepResult:
        """
        Exponential Euler,
        computes the solution update by:
        y_{t+1} = y_t + dt*\varphi_1(dt*L)F(t, y_t)
        """
        t = self.t
        yt = self.y_hist[0]

        cks = [(1.,[1])]
        phiLs = self._update_dense_phiLs(dt, cks)

        y_new, y_err = ExpSplitIntegrator._step_exp1_jit(t, yt, dt, self.sys, phiLs)

        return StepResult(t+dt, dt, y_new, y_err)

    @partial(jax.jit, static_argnames=('c2', ))
    def _step_exp2_jit(t, yt, dt, sys, sys_lop, phiLs, c2):
        print("jit-compiling exp2_dense kernel")
        fyt = sys.frhs(t, yt)
        # 2nd stage
        t_2 = t + c2*dt
        y_2 = yt + c2*dt*phiLs[(c2,1)] @ fyt
        r_2 = sys.frhs(t_2, y_2) - fyt - sys_lop(y_2 - yt)

        # compute final update
        y_new = yt + dt * (phiLs[(1.,1)] @ fyt + (1/c2) * phiLs[(1.,2)] @ r_2)

        # TODO: error estimate by comparing y_2 and y_new?
        y_err = -1.0
        return y_new, y_err

    def _step_exp2_dense(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        y_2 = y_t + c2*dt*\varphi_1(c2*dt*L)F(t, y_t)
        y_{t+1} = y_t + dt*\varphi_1(dt*L)F(t, y_t)
                      + dt/c2*\varphi_2(dt*L)R_2
        """
        t = self.t
        yt = self.y_hist[0]

        c2 = 2./3. #constant between 0 and 1
        cks = [(c2, [1]),
               (1., [1, 2])]
        phiLs = self._update_dense_phiLs(dt, cks)

        y_new, y_err = ExpSplitIntegrator._step_exp2_jit(t, yt, dt, self.sys, self._cached_sys_lop, phiLs, c2)

        return StepResult(t+dt, dt, y_new, y_err)

    @partial(jax.jit, static_argnames=('c2', 'c3'))
    def _step_exp3_jit(t, yt, dt, sys, sys_lop, phiLs, c2, c3):
        print("jit-compiling exp3_dense kernel")

        fyt = sys.frhs(t, yt)

        # 2nd stage
        t_2 = t + c2*dt
        y_2 = yt + c2*dt * phiLs[(c2,1)] @ fyt
        r_2 = sys.frhs(t_2, y_2) - fyt - sys_lop(y_2 - yt)

        # 3rd stage
        t_3 = t + c3*dt
        y_3 = yt + c3*dt * (phiLs[(c3,1)] @ fyt + (2./3./c2) * phiLs[(c3,2)] @ r_2)
        r_3 = sys.frhs(t_3, y_3) - fyt - sys_lop(y_3 - yt)

        # compute final update
        y_new = yt + dt * (phiLs[(1.,1)] @ fyt + (3./2.) * phiLs[(1.,2)] @ r_3)

        # TODO: error estimate?
        y_err = -1.0

        return y_new, y_err

    def _step_exp3_dense(self, dt: float) -> StepResult:
        """
        Computes the solution update by:
        y_2 = y_t + c2*dt*\varphi_1(c2*dt*L)F(t, y_t)
        y_3 = y_t + (2/3)*dt*\varphi_1((2/3)*dt*L)F(t, y_t) + (4./9./c2)*dt*\varphi_2((2/3)*dt*L)R_2
        y_{t+1} = y_t + dt*\varphi_1(dt*L)F(t, y_t) + 3/2*dt*\varphi_2(dt*L)R_3
        """
        t = self.t
        yt = self.y_hist[0]

        c2 = 2./3. # constant between 0 and 1
        c3 = 2./3. # == 2./3.
        cks = [(c2, [1]),
               (c3, [1, 2]),
               (1., [1, 2])]
        phiLs = self._update_dense_phiLs(dt, cks)

        y_new, y_err = ExpSplitIntegrator._step_exp3_jit(t, yt, dt, self.sys, self._cached_sys_lop, phiLs, c2, c3)

        return StepResult(t+dt, dt, y_new, y_err)

    def step(self, dt: float) -> StepResult:
        if self.method == "exp1":
            return self._step_exp1(dt)
        elif self.method == "exp2":
            return self._step_exp2(dt)
        elif self.method == "exp3":
            return self._step_exp3(dt)
        elif self.method == "exp1_dense":
            return self._step_exp1_dense(dt)
        elif self.method == "exp2_dense":
            return self._step_exp2_dense(dt)
        elif self.method == "exp3_dense":
            return self._step_exp3_dense(dt)
        else:
            raise NotImplementedError

    def accept_step(self, s: StepResult):
        self.t = s.t
        self.t_hist.appendleft(s.t)
        self.y_hist.appendleft(s.y)


