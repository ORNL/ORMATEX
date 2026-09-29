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
//! Exponential Rosenbrock (EXPRB) exponential integrators.
//!
//! Provides [`ExprbIntegrator`], an implementation of the third order
//! exponential Rosenbrock method EXPRB3 (also known as exprb32) with an
//! embedded second order error estimate, for the system $y^\prime  = f(t, y)$. The
//! solution is linearized about the current state $(t_n, y_n)$ as
//!
//! $$ f(t, y) = f_n + J_n (y - y_n) + (t - t_n) v_n + R_n(t, y) $$
//!
//! where $f_n = f(t_n, y_n)$, $J_n$ is the Jacobian, $v_n = \partial f / \partial t$
//! is the time derivative of the rhs (nonautonomous correction) and $R_n$ is the
//! nonlinear remainder. With $h = \Delta t$ and the phi-functions
//! $\varphi_k(z)$:
//!
//! EXPRB2 (exponential Rosenbrock-Euler, also the stage $Y_2$ below):
//!
//! $$ y_{n+1} = y_n + h \thinspace \varphi_1(h J_n) f_n + h^2 \varphi_2(h J_n) v_n $$
//!
//! EXPRB3, with second order stage $Y_2$ and third order solution $y_{n+1}$:
//!
//! $$ Y_2 = y_n + h \thinspace \varphi_1(h J_n) f_n + h^2 \varphi_2(h J_n) v_n $$
//!
//! $$ y_{n+1} = Y_2 + 2 h \thinspace \varphi_3(h J_n) R_n(t_n + h, Y_2) $$
//!
//! The $v_n$ terms are only included when the nonautonomous correction is
//! enabled and significant (see [`ExprbIntegrator::with_opt`]). The difference
//! $y_{n+1} - Y_2$ is used as the embedded error estimate (the norm used is
//! described in [`ExprbIntegrator`]), which can drive a step size controller
//! such as [`crate::ode_step_controller::BasicStepController`].
//!
//! The first stage is evaluated with a single matrix exponential action of an
//! extended operator ([`crate::ode_sys::DynRefExtendedLinOp`]) by the
//! phi-function evaluator ([`crate::matexp_traits::LinOpPhikvEvaluator`]).
//!
//! # References
//!
//! * Hochbruck, M., Ostermann, A. and Schweitzer, J., "Exponential Rosenbrock-type
//!   methods", SIAM J. Numer. Anal. 47(1) (2009) 786-803, doi:10.1137/080717717
//! * Hochbruck, M. and Ostermann, A., "Exponential integrators", Acta Numerica
//!   19 (2010) 209-286, doi:10.1017/S0962492910000048
use crate::matexp_traits::LinOpPhikvEvaluator;
use crate::ode_sys::*;
use crate::ode_traits::{IntegrateSys, StepperExponential};
use faer::prelude::*;
use std::collections::VecDeque;

/// Exponential Rosenbrock (EXPRB3) integrator with embedded error estimate.
///
/// Only the method `"exprb3"` is supported. See the module documentation for
/// the method formulas. Implements [`crate::ode_traits::IntegrateSys`]. Each
/// step returns an error estimate `err`, computed as the 1-norm of
/// $y_{n+1} - Y_2$, the difference between the third order solution and the
/// second order stage.
///
/// A mass matrix supplied by `OdeSys::fmass` is ignored.
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
    /// Set the initial conditions and setup the EXPRB integrator.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state
    /// * `method` - method name, must be `"exprb3"`
    /// * `expm` - phi-function evaluator used to compute the matrix
    ///   exponential and phi-function products
    ///
    /// # Panics
    ///
    /// Panics if `method` is not `"exprb3"`.
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
    ///
    /// Valid options:
    ///
    /// * `"tol_fdt"` - tolerance for the nonautonomous correction. If the value
    ///   is non-negative, the time derivative of the rhs, $v_n$, is estimated
    ///   with a forward finite difference (time step $10^{-8}$) and the
    ///   $h^2 \varphi_2(h J_n) v_n$ term is included in the update only if
    ///   $\Vert v_n\Vert_\infty$ exceeds `tol_fdt`. The default is `-1.0`
    ///   (disabled, $v_n = 0$).
    ///
    /// # Arguments
    ///
    /// * `option_str` - name of the option
    /// * `option_val` - value of the option
    ///
    /// # Panics
    ///
    /// Panics if `option_str` is not a valid option name.
    pub fn with_opt(mut self, option_str: String, option_val: f64) -> Self {
        match option_str.as_str() {
            "tol_fdt" => self.tol_fdt = option_val,
            _ => panic!("bad option"),
        };
        self
    }

    /// Exponential Rosenbrock order 3 with 2nd order embedded error estimate.
    ///
    /// $$ Y_2 = y_n + h \thinspace \varphi_1(h J_n) f_n + h^2 \varphi_2(h J_n) v_n $$
    /// $$ y_{n+1} = Y_2 + 2 h \thinspace \varphi_3(h J_n) R_n(t_n + h, Y_2) $$
    ///
    /// The error estimate is the 1-norm of $y_{n+1} - Y_2$.
    ///
    /// # References
    ///
    /// * Hochbruck, M., Ostermann, A. and Schweitzer, J., "Exponential
    ///   Rosenbrock-type methods", SIAM J. Numer. Anal. 47(1) (2009) 786-803
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
        let zero_n = faer::Mat::zeros(y0.nrows(), 1);

        // Compute time derivative of rhs if nonautonomous correction is ON
        let v = if self.tol_fdt < 0.0 {
            zero_n.clone()
        } else {
            self.frhs_fdt(sys, t, y0.as_ref(), fy0.as_ref(), 1e-8)
        };
        let vb2 = faer::Scale(dt.powi(2)) * v.as_ref();

        // build vector of rhs for phi functions
        let vb = if self.tol_fdt >= 0.0 && v.norm_max() > self.tol_fdt {
            vec![
                zero_n.as_ref(),
                fy0_dt.as_ref(),
                vb2.as_ref(),
            ]
        } else {
            vec![
                zero_n.as_ref(),
                fy0_dt.as_ref(),
            ]
        };

        // extended linear operator
        let ext_a_lo = DynRefExtendedLinOp::new(dt, sys_jac_lop.as_ref(), &vb);
        self.expm.apply_prepare(
            sys_jac_lop.as_ref(),
            dt,
            y0.as_ref(),
            2,
            Some((&ext_a_lo, &vb)),
        );

        // first stage
        let t_2 = t + dt;
        let y_2 = y0.as_ref()
            + self.expm.apply_phi_k_v(&ext_a_lo, 1.0, &vb);

        // nonlinear remainder
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

        // final stage
        let y_new = y_2.as_ref()
            + 2. * dt * self.expm.apply_phi_k(sys_jac_lop.as_ref(), dt, r_2.as_ref(), 3);

        // embedded error estimate
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
