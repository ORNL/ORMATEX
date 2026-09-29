/*
 * Copyright(c) 2025 UT-Battelle, LLC
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
//! Exponential Propagation Iterative Runge-Kutta (EPI) exponential integrators.
//!
//! Provides [`EpirkIntegrator`], an implementation of the EPI2 and EPI3
//! exponential time integration methods for the system $y^\prime  = f(t, y)$. The
//! solution is linearized about the current state $(t_n, y_n)$ as
//!
//! $$ f(t, y) = f_n + J_n (y - y_n) + (t - t_n) v_n + R_n(t, y) $$
//!
//! where $f_n = f(t_n, y_n)$, $J_n$ is the Jacobian, $v_n = \partial f / \partial t$
//! is the time derivative of the rhs (nonautonomous correction) and $R_n$ is the
//! nonlinear remainder. With $h = \Delta t$ and the phi-functions
//! $\varphi_k(z)$, the implemented methods are:
//!
//! EPI2 (exponential Euler, identical to EXPRB2):
//!
//! $$ y_{n+1} = y_n + h \thinspace \varphi_1(h J_n) f_n + h^2 \varphi_2(h J_n) v_n $$
//!
//! EPI3 (two-step, uses the previous state $y_{n-1}$ at $t_{n-1} = t_n - h$):
//!
//! $$ y_{n+1} = y_n + h \thinspace \varphi_1(h J_n) f_n +
//!   h^2 \varphi_2(h J_n) v_n +
//!   \frac{2}{3} h \thinspace \varphi_2(h J_n) R_n(t_{n-1}, y_{n-1}) $$
//!
//! with $R_n(t_{n-1}, y_{n-1}) = f(t_{n-1}, y_{n-1}) - f_n - J_n (y_{n-1} - y_n) -
//! (t_{n-1} - t_n) v_n$. The $v_n$ terms are only included when the
//! nonautonomous correction is enabled (see [`EpirkIntegrator::with_opt`]). The
//! EPI3 coefficient $2/3$ assumes a constant step size. The first EPI3 step,
//! for which no previous state is available, is taken with EPI2.
//!
//! The combination of phi-function products is evaluated with a single
//! matrix exponential action of an extended operator
//! ([`crate::ode_sys::DynRefExtendedLinOp`]) by the phi-function evaluator
//! ([`crate::matexp_traits::LinOpPhikvEvaluator`]). These methods provide no
//! embedded error estimate.
//!
//! # References
//!
//! * Tokman, M., "Efficient integration of large stiff systems of ODEs with
//!   exponential propagation iterative (EPI) methods", J. Comput. Phys. 213(2)
//!   (2006) 748-776, doi:10.1016/j.jcp.2005.08.032
//! * Gaudreault, S. and Pudykiewicz, J. A., "An efficient exponential time
//!   integration method for the numerical solution of the shallow water
//!   equations on the sphere", J. Comput. Phys. 322 (2016) 827-848
//! * Hochbruck, M. and Ostermann, A., "Exponential integrators", Acta Numerica
//!   19 (2010) 209-286, doi:10.1017/S0962492910000048
use crate::matexp_traits::LinOpPhikvEvaluator;
use crate::ode_sys::*;
use crate::ode_traits::{IntegrateSys, StepperExponential};
use faer::prelude::*;
use std::collections::VecDeque;

/// Exponential propagation iterative (EPI) integrator.
///
/// Supports the methods `"epi2"`, `"exprb2"` (both the exponential Euler method,
/// order 2) and `"epi3"` (order 3). See the module documentation for the
/// method formulas. Implements [`crate::ode_traits::IntegrateSys`]. Each step
/// prints timing information to stdout.
///
/// A mass matrix supplied by `OdeSys::fmass` is ignored.
pub struct EpirkIntegrator<T: LinOpPhikvEvaluator> {
    /// Matrix exponential evaluator
    expm: T,

    /// Order
    order: usize,

    /// Method
    method: String,

    /// Current time
    t: f64,

    /// tol used to check max derivative for nonautonomous system
    tol_fdt: f64,

    /// Storage for past system solution states
    y_hist: VecDeque<Mat<f64>>,
    t_hist: VecDeque<f64>,
}

impl<T> EpirkIntegrator<T>
where
    T: LinOpPhikvEvaluator,
{
    /// Set the initial conditions and setup the EPI integrator.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state
    /// * `method` - method name, one of `"epi2"`, `"exprb2"` or `"epi3"`
    /// * `expm` - phi-function evaluator used to compute the matrix
    ///   exponential and phi-function products
    ///
    /// # Panics
    ///
    /// Panics if `method` is not one of the valid method names.
    pub fn new(t0: f64, y0: MatRef<f64>, method: String, expm: T) -> Self {
        let order = match method.as_str() {
            "epi2" | "exprb2" => 2,
            "epi3" => 3,
            _ => panic!("invalid method: {:?}. Valid: epi2,epi3,exprb2", method),
        };
        let mut y_hist = VecDeque::with_capacity(order);
        let mut t_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        t_hist.push_front(t0);
        Self {
            expm,
            order,
            method,
            t: t0,
            tol_fdt: -1.0,
            y_hist,
            t_hist,
        }
    }

    /// Builder function to set optional solver parameters.
    ///
    /// Valid options:
    ///
    /// * `"tol_fdt"` - enables the nonautonomous correction if the value is
    ///   non-negative. The time derivative of the rhs, $v_n$, is then estimated
    ///   with a forward finite difference (time step $10^{-8}$) and included in
    ///   the update. The default is `-1.0` (disabled, $v_n = 0$). In this
    ///   integrator the value is only used as an on/off switch, it is not
    ///   compared against the size of $v_n$.
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

    /// Exponential Propagation Iterative order 2 method (EPI2)
    ///
    /// $$ y_{n+1} = y_n + h \thinspace \varphi_1(h J_n) f_n + h^2 \varphi_2(h J_n) v_n $$
    ///
    /// where the $v_n$ term is only present if the nonautonomous correction
    /// is enabled.
    ///
    /// # References
    ///
    /// * Gaudreault, S. and Pudykiewicz, J. A., "An efficient exponential time
    ///   integration method for the numerical solution of the shallow water
    ///   equations on the sphere", J. Comput. Phys. 322 (2016) 827-848
    /// * Tokman, M., "Efficient integration of large stiff systems of ODEs with
    ///   exponential propagation iterative (EPI) methods", J. Comput. Phys.
    ///   213(2) (2006) 748-776
    fn step_order_2<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: f64,
    ) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // setup jacobian linear operator evaluated at y0
        let sys_jac_lop = sys.fjac(t, y0.as_ref());
        let fy0 = sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);

        // correction for nonautonomous case
        let v: Mat<f64> = if self.tol_fdt < 0.0 {
            faer::Mat::zeros(y0.nrows(), 1)
        } else {
            self.frhs_fdt(sys, t, y0.as_ref(), fy0.as_ref(), 1e-8)
        };
        let vb2 = dt.powi(2) * v;

        // build vector of rhs
        let zero_mat = faer::Mat::zeros(y0.nrows(), 1);
        let vb = vec![zero_mat.as_ref(), fy0_dt.as_ref(), vb2.as_ref()];
        let ext_a_lo = DynRefExtendedLinOp::new(dt, sys_jac_lop.as_ref(), &vb);
        self.expm.apply_prepare(
            sys_jac_lop.as_ref(),
            dt,
            y0.as_ref(),
            2,
            Some((&ext_a_lo, &vb)),
        );
        let y_new = y0.as_ref() + self.expm.apply_phi_k_v(&ext_a_lo, 1.0, &vb);

        // return result
        Ok(StepResult::new(t + dt, dt, y_new, None))
    }

    /// Exponential Propagation Iterative order 3 method (EPI3)
    ///
    /// $$ y_{n+1} = y_n + h \thinspace \varphi_1(h J_n) f_n + h^2 \varphi_2(h J_n) v_n +
    ///   \frac{2}{3} h \thinspace \varphi_2(h J_n) R_n(t_{n-1}, y_{n-1}) $$
    ///
    /// Requires the previous state $y_{n-1}$ in the solution history and assumes
    /// it was computed with the same step size.
    ///
    /// # References
    ///
    /// * Gaudreault, S. and Pudykiewicz, J. A., "An efficient exponential time
    ///   integration method for the numerical solution of the shallow water
    ///   equations on the sphere", J. Comput. Phys. 322 (2016) 827-848
    fn step_order_3<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: f64,
    ) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let yp = self.y_hist[1].as_ref();
        let tp = self.t_hist[1];

        let sys_jac_lop = sys.fjac(t, y0.as_ref());
        let fy0 = sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);

        // correction for nonautonomous case
        let v: Mat<f64> = if self.tol_fdt < 0.0 {
            faer::Mat::zeros(y0.nrows(), 1)
        } else {
            self.frhs_fdt(sys, t, y0.as_ref(), fy0.as_ref(), 1e-8)
        };

        let rn_dt = faer::Scale(dt * 2.0 / 3.0)
            * self.remf(
                sys,
                t,
                y0.as_ref(),
                tp,
                yp.as_ref(),
                fy0.as_ref(),
                sys_jac_lop.as_ref(),
                Some(v.as_ref()),
            );
        let vb2 = rn_dt + dt.powi(2) * v;

        // build vector of rhs
        let zero_mat = faer::Mat::zeros(y0.nrows(), 1);
        let vb = vec![zero_mat.as_ref(), fy0_dt.as_ref(), vb2.as_ref()];
        let ext_a_lo = DynRefExtendedLinOp::new(dt, sys_jac_lop.as_ref(), &vb);
        self.expm.apply_prepare(
            sys_jac_lop.as_ref(),
            dt,
            y0.as_ref(),
            2,
            Some((&ext_a_lo, &vb)),
        );
        let y_new = y0.as_ref() + self.expm.apply_phi_k_v(&ext_a_lo, 1.0, &vb);

        // return result
        Ok(StepResult::new(t + dt, dt, y_new, None))
    }
}

impl<'a, T> IntegrateSys<'a> for EpirkIntegrator<T>
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
        println!("\nEPI step, t: {:?}, dt: {:?}", self.t, dt);
        let clock = std::time::Instant::now();
        let res = match self.method.as_str() {
            "epi2" | "exprb2" => self.step_order_2(sys, dt),
            "epi3" => {
                if self.y_hist.len() >= 2 {
                    self.step_order_3(sys, dt)
                } else {
                    self.step_order_2(sys, dt)
                }
            }
            _ => panic!("bad method"),
        };
        println!("EPI step time (s): {}", clock.elapsed().as_secs_f64());
        res
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
        self.t_hist.push_front(s.t);
        if self.y_hist.len() >= self.order + 1 {
            self.y_hist.pop_back();
            self.t_hist.pop_back();
        }
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.t_hist.clear();
        self.y_hist.push_front(y0.to_owned());
        self.t_hist.push_front(t0);
        self.t = t0;
    }
}

impl<T> StepperExponential for EpirkIntegrator<T> where T: LinOpPhikvEvaluator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matexp_krylov::KrylovExpm;
    use crate::matexp_pade::PadeExpm;
    use crate::test_common::TestLvSys;

    #[test]
    fn epi3_one_step() {
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0_f64], [4.0_f64]];
        let expm = KrylovExpm::new(Box::new(PadeExpm::new(12)), 4, 80, 1e-12, Some(2));
        let mut solver = EpirkIntegrator::new(0.0, y0.as_ref(), "epi3".to_string(), expm);

        let result = solver.step(&sys, 0.01).unwrap();
        assert_eq!(result.t, 0.01);
        assert!(result.err.is_none());
        solver.accept_step(result);
        assert_eq!(solver.time(), 0.01);
    }
}
