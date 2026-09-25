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
/// Exponential Rosenbrock class of exponential integrators.
use crate::matexp_traits::LinOpPhikvEvaluator;
use crate::ode_sys::*;
use crate::ode_traits::{IntegrateSys, StepperExponential};
use faer::prelude::*;
use std::collections::VecDeque;

pub struct ExprbIntegrator<T: LinOpPhikvEvaluator> {
    /// Matrix exponential evaluator
    expm: T,

    /// Current time
    t: f64,

    /// Tolerance used to check the maximum derivative for nonautonomous systems
    tol_fdt: f64,

    /// Current system solution state
    y_hist: VecDeque<Mat<f64>>,
}

impl<T> ExprbIntegrator<T>
where
    T: LinOpPhikvEvaluator,
{
    pub fn new(t0: f64, y0: MatRef<f64>, method: String, expm: T) -> Self {
        assert_eq!(method, "exprb3", "ExprbIntegrator only supports exprb3");
        let mut y_hist = VecDeque::with_capacity(1);
        y_hist.push_front(y0.to_owned());
        Self {
            expm,
            t: t0,
            tol_fdt: -1.0,
            y_hist,
        }
    }

    /// Builder function to set optional solver parameters.
    pub fn with_opt(mut self, option_str: String, option_val: f64) -> Self {
        match option_str.as_str() {
            "tol_fdt" => self.tol_fdt = option_val,
            _ => panic!("bad option"),
        };
        self
    }

    /// Exponential Rosenroack order 3 with 2nd order embedded error estimate.
    ///
    /// Ref: Hochbruck, Marlis, Alexander Ostermann, and Julia Schweitzer.
    /// Exponential Rosenbrock-type methods.
    /// SIAM Journal on Numerical Analysis 47.1 (2009): 786-803.
    ///
    fn step_exprb32<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: f64,
    ) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        let sys_jac_lop = sys.fjac(t, y0.as_ref());
        let fy0 = sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);

        // Use fy0 as a proxy Arnoldi starting vector until r_2 is available.
        let zero_n = faer::Mat::zeros(y0.nrows(), 1);
        let vb_prep = vec![
            zero_n.as_ref(),
            fy0_dt.as_ref(),
            fy0_dt.as_ref(),
            fy0_dt.as_ref(),
        ];
        let ext_a_prep = DynRefExtendedLinOp::new(dt, sys_jac_lop.as_ref(), &vb_prep);
        self.expm.apply_prepare(
            sys_jac_lop.as_ref(),
            dt,
            y0.as_ref(),
            3,
            Some((&ext_a_prep, &vb_prep)),
        );

        let v = if self.tol_fdt < 0.0 {
            Mat::zeros(y0.nrows(), 1)
        } else {
            self.frhs_fdt(sys, t, y0.as_ref(), fy0.as_ref(), 1e-8)
        };
        let phi2_v = if self.tol_fdt >= 0.0 && v.norm_max() > self.tol_fdt {
            faer::Scale(dt.powi(2))
                * self
                    .expm
                    .apply_phi_k(sys_jac_lop.as_ref(), dt, v.as_ref(), 2)
        } else {
            Mat::zeros(y0.nrows(), 1)
        };

        let t_2 = t + dt;
        let y_2 = y0.as_ref()
            + phi2_v.as_ref()
            + self
                .expm
                .apply_phi_k(sys_jac_lop.as_ref(), dt, fy0_dt.as_ref(), 1);
        let r_2 = self.remf(
            sys,
            t,
            y0.as_ref(),
            t_2,
            y_2.as_ref(),
            fy0.as_ref(),
            sys_jac_lop.as_ref(),
            Some(v.as_ref()),
        );

        let y_new = y_2.as_ref()
            + 2. * dt
                * self
                    .expm
                    .apply_phi_k(sys_jac_lop.as_ref(), dt, r_2.as_ref(), 3);
        let y_err = (y_new.as_ref() - y_2.as_ref()).as_ref().norm_l1().abs();

        Ok(StepResult::new(t + dt, dt, y_new, Some(y_err)))
    }
}

impl<T> StepperExponential for ExprbIntegrator<T> where T: LinOpPhikvEvaluator {}

impl<'a, T> IntegrateSys<'a> for ExprbIntegrator<T>
where
    T: LinOpPhikvEvaluator,
{
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        self.step_exprb32(sys, dt)
    }

    fn time(&self) -> Self::TimeType {
        self.t
    }

    fn state(&self) -> Self::SysStateType {
        self.y_hist[0].to_owned()
    }

    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>) {
        self.t = s.t;
        self.y_hist.push_front(s.y);
        self.y_hist.truncate(1);
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.y_hist.push_front(y0);
        self.t = t0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matexp_krylov::KrylovExpm;
    use crate::matexp_pade::PadeExpm;
    use crate::test_common::TestLvSys;

    #[test]
    fn exprb3_one_step() {
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0_f64], [4.0_f64]];
        let expm = KrylovExpm::new(Box::new(PadeExpm::new(12)), 4, 80, 1e-12, Some(2));
        let mut solver = ExprbIntegrator::new(0.0, y0.as_ref(), "exprb3".to_string(), expm);

        let result = solver.step(&sys, 0.01).unwrap();
        assert_eq!(result.t, 0.01);
        assert!(result.err.is_some());
        solver.accept_step(result);
        assert_eq!(solver.time(), 0.01);
    }
}
