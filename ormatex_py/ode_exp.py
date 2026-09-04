##############################################################################
# Copyright© 2025 UT-Battelle, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""
Exponential time integration methods
"""
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Callable

from ormatex_py.phi_evaluator import (
    PhiEvaluator,
    PhiPlan,
)

from ormatex_py.ode_sys import (
    LinOp,
    IntegrateSys,
    OdeSys,
    OdeSplitSys,
    StepResult,
)
from ormatex_py.matexp_krylov import phi_linop
from ormatex_py.matexp_phi import f_phi_k_sq_all, PhiEvaluator_PFD_Dense
from ormatex_py.matexp_leja import gen_leja_fast, gen_leja_conjugate, build_a_tilde, \
        leja_shift_scale, real_leja_expmv_substep, complex_conj_leja_expmv_substep


@IntegrateSys.populate_valid_methods
class ExpRBIntegrator(IntegrateSys):

    _deprecated_methods = {
        "exprb2_dense": "dense",
        "exprb2_pfd": "pfd",
        # "exprb3_pfd": "pfd", # TODO: add this and remove dedicated exprb3_pfd once not needed
        "exp_pfd": "pfd",
        "exprb2_pfd_rs": "pfd_rs",
        "exp_pfd_rs": "pdf_rs",
    }

    # initialize valid methods with deprecated versions
    # the actual non-deprecated methods are populated using the class decorator
    _valid_methods = IntegrateSys._valid_methods | _deprecated_methods

    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, method="epi2", phi_method=None, **kwargs):
        # Exponential integration method
        self.method = method
        if self.method not in self._valid_methods.keys():
            raise AttributeError(f"{self.method} not in {self._valid_methods}")

        # handle deprecated methods
        if self.method in self._deprecated_methods.keys():
            replacement_method = self.method.split("_", maxsplit=1)[0]
            self.phi_method = self._deprecated_methods[self.method]
            warnings.warn(f"'{self.method}' is deprecated, please use "
                          f"'{replacement_method}' with phi_method='{self.phi_method}'.")
            if phi_method is not None:
                warnings.warn(f"Using hardcoded phi_method '{self.phi_method}'"
                              f"for method '{self.method}' not {phi_method=}.")
            self.method = replacement_method
        else:
            # ensure that a phi_evaluator default is set
            self.phi_method = phi_method or self._valid_methods[self.method].default_phi

        # construct PhiEvaluator
        self.phi_method = self.phi_method or "krylov"
        self.Phi = PhiEvaluator(method=self.phi_method, sys_lop=None, **kwargs)

        order = self._valid_methods[self.method].order
        super().__init__(sys, t0, y0, order, self.method, **kwargs)

        # tolerence to detect nonautonomous systems, a negative value disables this check
        self.tol_fdt = kwargs.get("tol_fdt", 0.)

        # threads
        self.executor = ThreadPoolExecutor(max_workers=2)

    def reset_ic(self, t0: float, y0: jax.Array):
        super().reset_ic(t0, y0)

    @jax.jit(static_argnums=(4,))
    def _init_step(t: float, yt: jax.Array, sys, frhs_kwargs: dict, tol_fdt: float):
        """
        Computes inital linearizations and evaluations
        J_yt = d[frhs]/dy(t), fyt = frhs(t), fytt = d[frhs]/dt(t)
        """
        print("jit-compiling init_step kernel")
        sys_jac_lop = sys.fjac(t, yt, **frhs_kwargs)
        fyt = sys_jac_lop._frhs_cached()

        if tol_fdt >= 0.:
            # deriv of rhs wrt time at current time
            fytt = sys_jac_lop._fdt()
        else:
            fytt = jnp.zeros(fyt.shape)

        return sys_jac_lop, fyt, fytt

    @jax.jit
    def _remf(tr: float, yr: jax.Array,
              t: float, yt: jax.Array,
              fyt: jax.Array, sys_frhs: Callable, frhs_kwargs: dict,
              sys_jac_lop_yt: LinOp, fytt: jax.Array):
        """
        Computes remainder
        R(yr) = frhs(yr) - frhs(yt) - J_yt*(yr-yt) - fytt*(tr-t0)
        where fytt = d[frhs]/dt(t)
        """
        print("jit-compiling remf kernel")
        dt = tr - t
        fyr = sys_frhs(tr, yr, **frhs_kwargs)
        jac_yd = sys_jac_lop_yt(yr - yt)
        return fyr - fyt - jac_yd - fytt*dt

    @IntegrateSys.register_method(["exprb2", "epi2"], 2)
    def _step_exprb2(self, dt: float, frhs_kwargs: dict) -> StepResult:
        r"""
        Exponential Euler, computes the solution update by:

        .. math::

            y_{t+1} = y_t + dt*\varphi_1(dt*J)F(t, y_t) + dt**2*\varphi_2(dt*J)F'(t, y_t)

        where J is the Jacobian matrix and varphi is computed using a general PhiEvaluator

        Hochbruck, Marlis, Alexander Ostermann, and Julia Schweitzer.
        Exponential Rosenbrock-type methods.
        SIAM Journal on Numerical Analysis 47.1 (2009): 786-803.
        doi: https://doi.org/10.1137/080717717
        """
        t = self.t
        yt = self.y_hist[0]

        sys_jac_lop, fyt, fytt = ExpRBIntegrator._init_step(
            t, yt, self.sys, frhs_kwargs, self.tol_fdt)
        self.Phi.set_lop(sys_jac_lop)

        if self.tol_fdt >= 0. and jnp.linalg.norm(fytt, ord=jax.numpy.inf) > self.tol_fdt:
            # phi_plan = PhiPlan(dt, (dt,), ((1, 2),))
            y_update = self.Phi.eval_phis((1, 2), dt, (fyt, dt*fytt))
        else:
            # phi_plan = PhiPlan(dt, (dt,), ((1,),))
            y_update = self.Phi.eval_phi(1, dt, fyt)

        y_new = yt + dt * y_update
        # no error est. avail
        y_err = -1.

        return StepResult(t+dt, dt, y_new, y_err)

    @IntegrateSys.register_method(["epi3"], 2)
    def _step_epi3(self, dt: float, frhs_kwargs: dict) -> StepResult:
        """
        Exponential Propagation Integrator 3, computes the solution update by:

        .. math::

            y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t)
                    + (2/3)*dt*\varphi_2(dt*J_t)R(t, y_t, y_{t-1})
                    + dt**2*\varphi_2(dt*J_t)F'(t, y_t)

        Ref doi: https://doi.org/10.1137/110849961
        """
        t = self.t
        yt = self.y_hist[0] # y_t
        yp = self.y_hist[1] # y_{t-1}
        tp = self.t_hist[1]

        sys_jac_lop, fyt, fytt = ExpRBIntegrator._init_step(
            t, yt, self.sys, frhs_kwargs, self.tol_fdt)

        # residual
        # rn = self._remf(tp, yp, fyt, sys_jac_lop, fytt)
        rn = ExpRBIntegrator._remf(
            tp, yp, t, yt, fyt, self.sys.frhs, frhs_kwargs,
            sys_jac_lop, fytt)

        phi_plan = PhiPlan(dt, (dt,), ((1, 2),))
        phi_rhs = (fyt, (2./3.)*rn + dt*fytt)

        # hardcoded switch between different equivalent implementations
        use_plan = False
        if use_plan:
            self.Phi.set_lop(sys_jac_lop, phi_plan)
            y_update = self.Phi.eval_phis_plan(phi_rhs, step=0)
        else:
            self.Phi.set_lop(sys_jac_lop)
            y_update = self.Phi.eval_phis(phi_plan.ks[0], phi_plan.cdts[0], phi_rhs)

        y_new = yt + dt * y_update
        # no error est. avail
        y_err = -1.0

        return StepResult(t+dt, dt, y_new, y_err)

    @IntegrateSys.register_method(["exprb3"], 3)
    def _step_exprb3(self, dt: float, frhs_kwargs: dict) -> StepResult:
        """
        Exponential Rosenbrock 3, computes the solution update by:

        .. math::

            y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t)
                    + 2*dt*\varphi_3(dt*J_t)R_2
                    + dt**2*\varphi_2(dt*J_t)F'(t, y_t)

        Hochbruck, Marlis, Alexander Ostermann, and Julia Schweitzer.
        Exponential Rosenbrock-type methods.
        SIAM Journal on Numerical Analysis 47.1 (2009): 786-803.
        doi: https://doi.org/10.1137/080717717
        """
        t = self.t
        yt = self.y_hist[0]

        sys_jac_lop, fyt, fytt = ExpRBIntegrator._init_step(
            t, yt, self.sys, frhs_kwargs, self.tol_fdt)

        if self.tol_fdt >= 0. and jnp.linalg.norm(fytt, ord=jax.numpy.inf) > self.tol_fdt:
            phi_plan = PhiPlan(dt, (dt, dt), ((1, 2), (3,)))
            phi_rhs_0 = (fyt, dt*fytt)
        else:
            phi_plan = PhiPlan(dt, (dt, dt), ((1,), (3,)))
            phi_rhs_0 = (fyt,)

        self.Phi.set_lop(sys_jac_lop, phi_plan)

        # 2nd stage
        y_update1 = self.Phi.eval_phis_plan(phi_rhs_0, step=0)
        t_2 = t + dt
        y_2 = yt + dt * y_update1

        r_2 = ExpRBIntegrator._remf(
            t_2, y_2, t, yt, fyt, self.sys.frhs, frhs_kwargs, sys_jac_lop, fytt)

        # compute final update
        y_update2 = self.Phi.eval_phis_plan((2.*r_2,), step=1)
        y_new = y_2 + dt * y_update2

        # error estimate by comparing y_2 and y_new
        y_err = jnp.linalg.norm(y_2 - y_new, ord=jnp.inf)

        return StepResult(t+dt, dt, y_new, y_err)

    @IntegrateSys.register_method(["exprb4", "pexprb4"], 4)
    def _step_pexprb4(self, dt: float, frhs_kwargs: dict) -> StepResult:
        r"""
        Computes the solution update by:

        .. math::

            U_{n2} = u_n + (0.5)dt\varphi_1((0.5)dt J_t)F(t, y_t) +
                0.5^2 dt^2 \varphi_2(0.5 dt*J_t)F'(t, y_t)
            U_{n3} = u_n + (1.0)dt\varphi_1((1.0)dt J_t)F(t, y_t) +
                dt^2 \varphi_2(dt*J_t)F'(t, y_t)

            y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t) +
                dt^2 \varphi_2(dt*J_t)F'(t, y_t) +
                dt [16 \varphi_3(dt J_t) - 48 \varphi_4(dt J_t)] R_2(U_{n2}) +
                dt [-2 \varphi_3(dt J_t) + 12 \varphi_4(dt J_t)] R_3(U_{n3})

        Where $`U_{n2}`$ and $`U_{n3}`$ can be computed in parallel.

        Ref: V. T. Luan. and A. Ostermann. Parallel Exponential Rosenbrock Methods.
        Computers and Mathmatics with Applications, v71. 2016.
        """
        t = self.t
        yt = self.y_hist[0] # y_t

        sys_jac_lop, fyt, fytt = ExpRBIntegrator._init_step(
            t, yt, self.sys, frhs_kwargs, self.tol_fdt)

        have_fytt = False
        if self.tol_fdt >= 0.:
            have_fytt = jnp.linalg.norm(fytt, ord=jax.numpy.inf) > self.tol_fdt

        # butcher tableau coeffs
        c_2 = 0.5
        c_3 = 1.0

        # build plan
        if have_fytt:
            phi_plan = PhiPlan(dt,
                               (c_2*dt, c_3*dt, dt),
                               ((1, 2), (1, 2), (3, 4)))
        else:
            phi_plan = PhiPlan(dt,
                               (c_2*dt, c_3*dt, dt),
                               ((1,), (1,), (3, 4)))

        self.Phi.set_lop(sys_jac_lop, phi_plan)

        # compute U_{n2}
        def f_u_n2():
            t_2 = t + c_2*dt
            if have_fytt:
                phi_rhs = (c_2*fyt, c_2**2*dt*fytt)
            else:
                phi_rhs = (c_2*fyt,)
            y_2 = yt + dt * self.Phi.eval_phis_plan(phi_rhs, step=0)
            r_2 = ExpRBIntegrator._remf(
                t_2, y_2, t, yt, fyt, self.sys.frhs, frhs_kwargs, sys_jac_lop, fytt)
            return r_2

        # compute U_{n3}
        def f_u_n3():
            t_3 = t + c_3*dt
            if have_fytt:
                phi_rhs = (c_3*fyt, c_3**2*dt*fytt)
            else:
                phi_rhs = (c_3*fyt,)
            y_3 = yt + dt * self.Phi.eval_phis_plan(phi_rhs, step=1)
            r_3 = ExpRBIntegrator._remf(
                t_3, y_3, t, yt, fyt, self.sys.frhs, frhs_kwargs, sys_jac_lop, fytt)
            return y_3, r_3

        # with ThreadPoolExecutor(max_workers=2) as executor:
        fut_f_u_n2 = self.executor.submit(f_u_n2)
        fut_f_u_n3 = self.executor.submit(f_u_n3)
        r_2 = fut_f_u_n2.result()
        y_3, r_3 = fut_f_u_n3.result()

        # compute final update
        phi_rhs_3 = (16.0*r_2-2.0*r_3, -48.0*r_2+12.0*r_3)
        # b2_b3 = self.Phi.eval_phis((3, 4), dt, phi_rhs_3)
        b2_b3 = self.Phi.eval_phis_plan(phi_rhs_3, step=2)
        y_new = y_3 + dt * b2_b3

        # Estimate the step error
        # Note: this is the difference between the 4th order estimate and a 2nd order est.
        y_err = jnp.linalg.norm(y_3 - y_new, ord=jnp.inf)
        return StepResult(t+dt, dt, y_new, y_err)

    @IntegrateSys.register_method(["exprb3_pfd"], 3)
    def _step_exprb3_pfd(self, dt: float, frhs_kwargs: dict) -> StepResult:
        r"""
        Computes the solution update by:

        .. math::

            y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t) +
                2*dt*\varphi_3(dt*J_t)R_2 +
                dt**2*\varphi_2(dt*J_t)F'(t, y_t)

        Using a partial fraction decomposition for $`\varphi`$.
        """
        t = self.t
        yt = self.y_hist[0] # y_t

        sys_jac_lop, fyt, fytt = ExpRBIntegrator._init_step(
            t, yt, self.sys, frhs_kwargs, self.tol_fdt)

        J = sys_jac_lop.dense()
        Jdt = dt*J

        # precompute LU decomposition of scaled jacobian for each pole, p
        # LU(J*dt-p*I)
        pfd_lu = PhiEvaluator_PFD_Dense(Jdt, "cram_16")

        # 1st stage
        # prepare PhiEvaluator arguments
        s1_k = (1,)         # Phi function order
        s1_b = (dt * fyt,)  # Phi RHS

        # check for nonautonomous system
        if self.tol_fdt >= 0.:
            # deriv of rhs wrt time at current time
            if jnp.linalg.norm(fytt, ord=jax.numpy.inf) > self.tol_fdt:
                # build additional RHS
                s1_k = (1, 2,)
                s1_b = (s1_b[0], (dt**2.0)*fytt,)

        s1_update = pfd_lu.apply(jnp.stack(s1_b, axis=1), jnp.array(s1_k))
        y_2 = yt + jnp.sum(s1_update, axis=1)

        # 2nd stage
        t_2 = t + dt
        # r_2 = self._remf(t_2, y_2, fyt, sys_jac_lop, fytt)
        r_2 = ExpRBIntegrator._remf(
            t_2, y_2, t, yt, fyt, self.sys.frhs, frhs_kwargs, sys_jac_lop, fytt)
        s2_k, s2_b = (3,), (2.0*dt*r_2,)
        s2_update = pfd_lu.apply(jnp.stack(s2_b, axis=1), jnp.array(s2_k))

        # compute final update
        y_new = y_2 + jnp.sum(s2_update, axis=1)
        y_err = jnp.linalg.norm(y_2 - y_new, ord=jnp.inf)

        return StepResult(t+dt, dt, y_new, y_err)

    @IntegrateSys.register_method(["exp"], 1)
    def _step_exp(self, dt: float, frhs_kwargs: dict) -> StepResult:
        r"""
        Computes the solution update by:
            y_{t+1} = \varphi_0(dt*J)*y0
        where J is the Jacobian matrix
        NOTE: Only useful for purely linear systems
        """
        t = self.t
        yt = self.y_hist[0]

        sys_jac_lop = self.sys.fjac(t, yt)
        self.Phi.set_lop(sys_jac_lop)

        fyt = sys_jac_lop._frhs_cached()
        if jnp.linalg.norm(fyt - sys_jac_lop(yt), ord=jax.numpy.inf) / jnp.linalg.norm(fyt, ord=jax.numpy.inf) > 1e-7:
            warnings.warn("Method step_exp is only valid for purely linear systems")

        y_new = self.Phi.eval_phi(0, dt, yt)

        y_err = -1.
        return StepResult(t+dt, dt, y_new, y_err)

    def step(self, dt: float, frhs_kwargs: dict = {}) -> StepResult:
        if len(self.y_hist) < 2 and self.method == "epi3":
            return self._valid_methods["exprb3"](self, dt, frhs_kwargs)
        else:
            return self._valid_methods[self.method](self, dt, frhs_kwargs)


@IntegrateSys.populate_valid_methods
class ExpLejaIntegrator(IntegrateSys):

    # new class list of valid methods
    _valid_methods = {}

    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, method="epi2_leja", **kwargs):
        # Exponential integration method
        self.method = method
        if method not in self._valid_methods.keys():
            raise AttributeError(f"{self.method} not in {self._valid_methods}")
        order = self._valid_methods[self.method].order
        # Relative tol for leja polynomial approx
        self.leja_tol = kwargs.get("leja_tol", 1e-15)
        # Optional max magnitude of real component of eigs(J*dt)
        self.leja_a = kwargs.get("leja_a", None)
        # Optional max magnitude of imag component of eigs(J*dt)
        self.leja_c = kwargs.get("leja_c", None)
        # Option to enable substepping
        self.leja_substep = kwargs.get("leja_substep", True)
        # Initial substep size
        self.leja_substep_size = 1.0
        # Method used to compute divided diffs
        self.dd_method = kwargs.get("dd_method", "taylor")
        # number of repeated zeros prepended to the leja sequence
        self.leja_n_zeros = int(kwargs.get("leja_n_zeros", 2))
        # eigenvector corrosponding to larget magnitude eigenvalue of sys jac.
        self._leja_bk = None
        self.istep = 0
        self.leja_max_power_iter = 100
        self.leja_max_re_eig_scale = kwargs.get("leja_max_re_eig_scale", 1.2)
        self.n_leja = kwargs.get("n_leja", 280)
        self.leja_x = jnp.asarray(
                gen_leja_fast(a=-2, b=2, n=self.n_leja))
        super().__init__(sys, t0, y0, order, method, **kwargs)

    @IntegrateSys.register_method(["epi2_leja_re"], 2)
    def _step_epi_re(self, dt: float, frhs_kwargs: dict) -> StepResult:
        """
        Computes the solution update by:

        .. math::

            y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t) +
                dt**2*\varphi_2(dt*J_t)F'(t, y_t)

        using the real leja point method (ReLPM).

        Ref:
            M. Caliari, M. Vianello, L. Bergamaschi.
            Interpolating discrete advection–diffusion propagators at Leja sequences.
            Journal of Computational and Applied Mathematics.
            Volume 172, Issue 1.
        """
        t = self.t
        yt = self.y_hist[0] # y_t

        sys_jac_lop = self.sys.fjac(t, yt, frhs_kwargs=frhs_kwargs)
        fyt = sys_jac_lop._frhs_cached()

        # time derivative
        fytt = sys_jac_lop._fdt()

        vb0 = jnp.zeros(yt.shape)

        # build augmented linear system
        a_tilde_lo, v, n = build_a_tilde(sys_jac_lop, dt, [vb0, dt*fyt, dt**2*fytt])

        if self.leja_a is None:
            # estimate largest magnitude eigenvalue and corrosponding eigenvec
            # by power iter.  Store eigenvector for next step
            # to speed convergence of power iterations in
            # subsequent calls to power iter method.
            shift, scale, max_eig, self._leja_bk, _power_iters = leja_shift_scale(
                    a_tilde_lo, v.shape[0], self.leja_max_power_iter,
                    self._leja_bk, self.leja_max_re_eig_scale)
        else:
            shift = dt*self.leja_a / 2.
            scale = np.abs(dt*self.leja_a / 4.)

        # compute phi-vector products by leja interpolation
        y_update, leja_iters, converged, max_tau_dt = real_leja_expmv_substep(
                a_tilde_lo, 1.1*self.leja_substep_size, v, self.leja_x,
                n, shift, scale, self.leja_tol, self.leja_substep,
                dd_method=self.dd_method)

        print("=== Total leja iters: %d, shift: %0.3f, scale: %0.3f" % (leja_iters, shift, scale))

        if not converged:
            raise RuntimeError("Leja not converged")

        y_new = yt + y_update
        y_err = -1.0
        self.istep += 1

        wgt = min(self.istep, 10)
        self.leja_substep_size = (1. / wgt) * max_tau_dt + \
                ((wgt-1) / wgt) * self.leja_substep_size

        return StepResult(t+dt, dt, y_new, y_err)

    @IntegrateSys.register_method(["epi2_leja_im", "epi3_leja_im"], 3)
    def _step_epi_im(self, dt: float, frhs_kwargs: dict, order: int = 2) -> StepResult:
        """
        Computes the solution update by:

        .. math::

            y_{t+1} = y_t + dt*\varphi_1(dt*J_t)F(t, y_t) +
                dt**2*\varphi_2(dt*J_t)F'(t, y_t)

        using a complex conjugate Leja point method (CLaPM).
        """
        t = self.t
        yt = self.y_hist[0] # y_t

        sys_jac_lop = self.sys.fjac(t, yt, frhs_kwargs=frhs_kwargs)
        fyt = sys_jac_lop._frhs_cached()

        # time derivative
        fytt = sys_jac_lop._fdt()

        vb0 = jnp.zeros(yt.shape)

        # build augmented linear system
        if order == 2:
            a_tilde_lo, v, n = build_a_tilde(sys_jac_lop, dt, [vb0, dt*fyt, dt**2*fytt])
        elif order == 3:
            yp = self.y_hist[1] # y_{t-1}
            tp = self.t_hist[1]
            rn = self._remf(tp, yp, fyt, sys_jac_lop, fytt)
            a_tilde_lo, v, n = build_a_tilde(sys_jac_lop, dt, [vb0, dt*fyt, dt*(2./3.)*rn + dt**2*fytt])
        else:
            raise NotImplementedError

        _power_iters = 0
        if self.leja_a is None:
            # estimate largest magnitude eigenvalue and corrosponding eigenvec
            # by power iter.  Store eigenvector for next step
            # to speed convergence of power iterations in
            # subsequent calls to power iter method.
            _, _, max_eig, self._leja_bk, _power_iters = leja_shift_scale(
                    a_tilde_lo, v.shape[0], self.leja_max_power_iter,
                    self._leja_bk, self.leja_max_re_eig_scale)
            leja_a = -jnp.abs(max_eig)
        else:
            leja_a = self.leja_a * dt
        if self.leja_c is None:
            leja_c = 0.0
        else:
            leja_c = self.leja_c * dt

        # generate leja sequence on the ellipse bounding the spectrum of the sys Jacobian
        leja_x, n_leja_real, scale, shift = gen_leja_conjugate(n=self.n_leja, a=leja_a, b=0., c=leja_c)
        leja_x = jnp.asarray(leja_x)

        # compute phi-vector products by leja interpolation
        y_update, leja_iters, converged, max_tau_dt = complex_conj_leja_expmv_substep(
                a_tilde_lo, 1.1*self.leja_substep_size, v, leja_x, n_leja_real,
                n, shift, scale, self.leja_tol, self.leja_substep,
                leja_n_zeros=self.leja_n_zeros, dd_method=self.dd_method)

        print("=t: %0.2f, Pwr itrs: %d, Leja itrs: %d, leja_a: %0.2f, leja_c: %0.2f, shift: %0.2f, scale: %0.2f" % (t, _power_iters, leja_iters, leja_a, leja_c, shift, scale))

        if not converged:
            raise RuntimeError("Leja not converged")

        y_new = yt + y_update
        y_err = -1.0
        self.istep += 1

        wgt = min(self.istep, 10)
        self.leja_substep_size = (1. / wgt) * max_tau_dt + \
                ((wgt-1) / wgt) * self.leja_substep_size

        return StepResult(t+dt, dt, y_new, y_err)

    def step(self, dt: float, frhs_kwargs: dict = {}) -> StepResult:
        if self.method == "epi3_leja_im":
            if len(self.y_hist) >= 2:
                return self._step_epi_im(dt, frhs_kwargs, order=3)
            else:
                return self._step_epi_im(dt, frhs_kwargs, order=2)
        else:
            return self._valid_methods[self.method](self, dt, frhs_kwargs)


@IntegrateSys.populate_valid_methods
class ExpSplitIntegrator(IntegrateSys):

    # new class list of valid methods
    _valid_methods = {}

    def __init__(self, sys: OdeSplitSys, t0: float, y0: jax.Array, method="exp3", **kwargs):
        self.method = method
        if not self.method in self._valid_methods.keys():
            raise AttributeError
        order = self._valid_methods[self.method].order
        super().__init__(sys, t0, y0, order, method, **kwargs)

        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        self.iom = kwargs.get("iom", 200)

        # TODO: used to check if dense phiL matrices need updating
        self._cached_dt = None

    def reset_ic(self, t0: float, y0: jax.Array):
        super().reset_ic(t0, y0)
        self._cached_dt = None

    def _remf(self, tr: float, yr: jax.Array,
              frhs_yt: jax.Array, sys_lop: LinOp):
        """
        Computes remainder R(yr) = frhs(yr) - frhs(yt) - L*(yr-yt)
        """
        yt = self.y_hist[0]
        frhs_yr = self.sys.frhs(tr, yr)
        L_yd = sys_lop(yr - yt)
        return frhs_yr - frhs_yt - L_yd

    @IntegrateSys.register_method(["exp1"], 1)
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

    @IntegrateSys.register_method(["exp2"], 2)
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

    @IntegrateSys.register_method(["exp3"], 3)
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
            # TODO: L is only evaluated the first step (t0, y0) or if dt changes.
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

    @IntegrateSys.register_method(["exp1_dense"], 1)
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

    @IntegrateSys.register_method(["exp2_dense"], 2)
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

    @IntegrateSys.register_method(["exp3_dense"], 3)
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

    def step(self, dt: float, frhs_kwargs: dict = {}) -> StepResult:
        return self._valid_methods[self.method](self, dt)
