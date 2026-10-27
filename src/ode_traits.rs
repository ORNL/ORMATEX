/*
 * Copyright(c) 2025,2026 UT-Battelle, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//! Common traits implemented by the ODE time integrators.
//!
//! This module defines [`IntegrateSys`], the interface shared by all time
//! integrators (step, accept, reset), and [`StepperExponential`], a helper trait
//! that provides the finite difference time derivative and the nonlinear
//! remainder used by exponential integrators such as EPI and EXPRB methods.
//!
//! # References
//!
//! * Hochbruck, M. and Ostermann, A., "Exponential integrators",
//!   Acta Numerica 19 (2010) 209-286, doi:10.1017/S0962492910000048
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::prelude::*;

use crate::ode_sys::{OdeSys, StepError, StepResult};

/// Common interface for ODE integrators.
///
/// An integrator holds the current time and state (and any history needed by
/// multistep methods). A step is first proposed with [`IntegrateSys::step`]
/// and only recorded once it is passed to [`IntegrateSys::accept_step`], which
/// allows a step size controller to reject a step and retry with a smaller `dt`.
pub trait IntegrateSys<'a> {
    /// Type used to represent time (typically `f64`)
    type TimeType;
    /// Type used to represent the system state (typically `faer::Mat<f64>`)
    type SysStateType;

    /// Step solution forward by `dt`, proposes a new state.
    ///
    /// This does not modify the recorded solution history. It may outright fail
    /// due to a numerical issue.
    ///
    /// # Arguments
    ///
    /// * `sys` - the ODE system to integrate
    /// * `dt` - time step size
    ///
    /// # Returns
    ///
    /// The proposed [`StepResult`] (new time, state and optional error estimate).
    ///
    /// # Errors
    ///
    /// Returns a [`StepError`] if the step fails.
    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError>;

    /// Get current (accepted) time.
    fn time(&self) -> Self::TimeType;

    /// Get current (accepted) system state.
    fn state(&self) -> Self::SysStateType;

    /// Accepts the proposed new time and state.
    ///
    /// Records accepted state into solution history.
    ///
    /// # Arguments
    ///
    /// * `s` - a step result previously returned by [`IntegrateSys::step`]
    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>);

    /// Reset integrator. Removes solution history.
    ///
    /// # Arguments
    ///
    /// * `t0` - new initial time
    /// * `y0` - new initial state
    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType);
}

/// Trait for exponential time integration implementors.
///
/// Provides default helper methods shared by exponential integrators, which
/// linearize the system about the current state $y_0$ as
///
/// $$ f(t, y) = f(t_0, y_0) + J (y - y_0) + (t - t_0) v + R(t, y) $$
///
/// where $J$ is the Jacobian, $v = \partial f / \partial t$ is the time
/// derivative of the rhs (nonautonomous correction) and $R$ is the nonlinear
/// remainder.
pub trait StepperExponential {
    /// Estimate the time derivative of the RHS by a forward finite difference.
    ///
    /// Computes
    ///
    /// $$ v \approx \frac{f(t_0 + \delta t, y_0) - f(t_0, y_0)}{\delta t} $$
    ///
    /// # Arguments
    ///
    /// * `sys` - the ODE system
    /// * `t0` - current time
    /// * `y0` - current state
    /// * `frhs_y0` - the rhs already evaluated at $(t_0, y_0)$
    /// * `del_t` - finite difference step in time, $\delta t$
    ///
    /// # Returns
    ///
    /// The estimate of $\partial f / \partial t$ at $(t_0, y_0)$.
    fn frhs_fdt(
        &self,
        sys: &dyn OdeSys<'_>,
        t0: f64,
        y0: MatRef<f64>,
        frhs_y0: MatRef<f64>,
        del_t: f64,
    ) -> Mat<f64> {
        let frhs_t1 = sys.frhs(t0 + del_t, y0);
        (frhs_t1 - frhs_y0) / Scale(del_t)
    }

    /// Compute the nonlinear remainder of the rhs at a point $(t_r, y_r)$.
    ///
    /// Evaluates
    ///
    /// $$ R(t_r, y_r) = f(t_r, y_r) - f(t_0, y_0) - J (y_r - y_0) - (t_r - t_0) v $$
    ///
    /// where $J$ is the Jacobian at $(t_0, y_0)$ and $v$ is the (optional)
    /// time derivative of the rhs. This is the quantity appended to the
    /// $\varphi_k$ terms in EPI and EXPRB methods.
    ///
    /// # Arguments
    ///
    /// * `sys` - the ODE system
    /// * `t0` - linearization time
    /// * `y0` - linearization state
    /// * `tr` - time at which to evaluate the remainder
    /// * `yr` - state at which to evaluate the remainder
    /// * `frhs_y0` - the rhs already evaluated at $(t_0, y_0)$
    /// * `sys_jac_lop_y0` - the Jacobian operator $J$ at $(t_0, y_0)$
    /// * `v` - optional time derivative of the rhs $v$; `None` is treated as zero
    ///
    /// # Returns
    ///
    /// The remainder $R(t_r, y_r)$.
    fn remf<'b>(
        &self,
        sys: &'b dyn OdeSys<'b>,
        t0: f64,
        y0: MatRef<f64>,
        tr: f64,
        yr: MatRef<f64>,
        frhs_y0: MatRef<f64>,
        sys_jac_lop_y0: &dyn LinOp<f64>,
        v: Option<MatRef<f64>>,
    ) -> Mat<f64> {
        let frhs_yr = sys.frhs(tr, yr);

        let yd = yr.as_ref() - y0.as_ref();
        let par = faer::get_global_parallelism();
        let mut buffer = MemBuffer::new(sys_jac_lop_y0.apply_scratch(yd.ncols(), par));
        let mut jac_yd = faer::Mat::zeros(sys_jac_lop_y0.nrows(), yd.ncols());
        sys_jac_lop_y0.apply(
            jac_yd.as_mut(),
            yd.as_ref(),
            par,
            MemStack::new(&mut buffer),
        );

        let dt = tr - t0;
        let vn_t = match v {
            Some(v) => Scale(dt) * v,
            None => Scale(dt) * Mat::<f64>::zeros(yr.nrows(), yr.ncols()).as_ref(),
        };
        frhs_yr - frhs_y0 - jac_yd - vn_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode_sys::get_fd_jac;
    use faer::matrix_free::LinOp;

    struct TimeScaledSys;

    impl<'a> OdeSys<'a> for TimeScaledSys {
        fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
            Scale(t) * x
        }

        fn fjac<'b>(&'a self, t: f64, x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
            Box::new(get_fd_jac(self, t, x))
        }
    }

    struct TestStepper;

    impl StepperExponential for TestStepper {}

    #[test]
    fn remf_with_and_without_time_correction() {
        let sys = TimeScaledSys;
        let stepper = TestStepper;
        let y0 = faer::mat![[2.0_f64], [3.0]];
        let yr = faer::mat![[2.5_f64], [4.0]];
        let jac = Mat::<f64>::identity(2, 2);
        let without = stepper.remf(
            &sys,
            1.0,
            y0.as_ref(),
            1.5,
            yr.as_ref(),
            y0.as_ref(),
            &jac,
            None,
        );
        let with = stepper.remf(
            &sys,
            1.0,
            y0.as_ref(),
            1.5,
            yr.as_ref(),
            y0.as_ref(),
            &jac,
            Some(y0.as_ref()),
        );
        assert!((without - Scale(0.5) * yr.as_ref()).norm_max() < 1e-12);
        assert!((with - Scale(0.5) * (yr - y0)).norm_max() < 1e-12);
    }

    #[test]
    fn frhs_fdt_estimates_time_derivative() {
        let sys = TimeScaledSys;
        let stepper = TestStepper;
        let y0 = faer::mat![[2.0_f64], [3.0_f64]];
        let fy0 = sys.frhs(1.0, y0.as_ref());

        let fdt = stepper.frhs_fdt(&sys, 1.0, y0.as_ref(), fy0.as_ref(), 1e-8);

        assert!((fdt - y0.as_ref()).norm_max() < 1e-6);
    }
}
