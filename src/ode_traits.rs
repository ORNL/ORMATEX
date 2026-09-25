/*
 * Copyright© 2025,2026 UT-Battelle, LLC
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
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;

use crate::ode_sys::{OdeSys, StepError, StepResult};

/// Common interface for ODE integrators.
pub trait IntegrateSys<'a> {
    type TimeType;
    type SysStateType;

    /// Step solution forward by dt, proposes a new state.
    /// This may outright fail due to numerical issue
    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError>;

    /// Get current time
    fn time(&self) -> Self::TimeType;

    /// Get current system state
    fn state(&self) -> Self::SysStateType;

    /// Accepts the proposed new time and state.
    /// Records accepted state into solution history.
    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>);

    /// Reset integrator. Removes solution history
    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType);
}

/// Trait for exponential time integration implementors
pub trait StepperExponential {
    /// Estimate the time derivative of the RHS by a forward finite difference.
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

        let mut jac_yd = faer::Mat::zeros(y0.nrows(), 1);
        sys_jac_lop_y0.apply(
            jac_yd.as_mut(),
            (yr.as_ref() - y0.as_ref()).as_ref(),
            faer::get_global_parallelism(),
            MemStack::new(&mut MemBuffer::new(StackReq::empty())),
        );

        let dt = tr - t0;
        let vn_t = Scale(dt) * v.unwrap_or(Mat::zeros(yr.nrows(), yr.ncols()).as_ref());
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
    fn frhs_fdt_estimates_time_derivative() {
        let sys = TimeScaledSys;
        let stepper = TestStepper;
        let y0 = faer::mat![[2.0_f64], [3.0_f64]];
        let fy0 = sys.frhs(1.0, y0.as_ref());

        let fdt = stepper.frhs_fdt(&sys, 1.0, y0.as_ref(), fy0.as_ref(), 1e-8);

        assert!((fdt - y0.as_ref()).norm_max() < 1e-6);
    }
}
