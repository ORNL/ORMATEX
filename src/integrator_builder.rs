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
//! Builders for ORMATEX time integrators.
//!
//! This module provides validated, typed builders that construct boxed
//! integrators ([`BuiltIntegrator`]) for solving $y^\prime  = f(t, y)$. There is one
//! builder per method family:
//!
//! * [`ExplicitIntegratorBuilder`] with [`ExplicitMethod`]: explicit Runge-Kutta
//!   methods of order 1 to 4.
//! * [`ImplicitIntegratorBuilder`] with [`ImplicitMethod`]: BDF and DIRK/SDIRK
//!   methods that solve stage equations with a Newton-Krylov iteration.
//! * [`ExponentialIntegratorBuilder`] with [`ExponentialMethod`]: exponential
//!   time integrators (EPI and exponential Rosenbrock) whose
//!   $\varphi_k$-function vector products are computed by an
//!   [`ExponentialEvaluator`] (Krylov, Leja, or Taylor; see [`KrylovOptions`],
//!   [`LejaOptions`], [`TaylorOptions`]).
//!
//! Each builder validates its options in `build()` and reports problems as an
//! [`IntegratorBuildError`] instead of panicking. Method names can be parsed from
//! strings with `FromStr` (case insensitive), which is used by the python
//! bindings in `ormatex_rspy`.
//!
//! # References
//!
//! * Hairer, E., Norsett, S. P., Wanner, G. Solving Ordinary Differential
//!   Equations I: Nonstiff Problems. Springer, 1993.
//! * Hairer, E., Wanner, G. Solving Ordinary Differential Equations II: Stiff
//!   and Differential-Algebraic Problems. Springer, 1996.
//! * Hochbruck, M., Ostermann, A. Exponential integrators. Acta Numerica 19
//!   (2010) 209-286. doi:10.1017/S0962492910000048
mod exponential;
mod explicit;
mod implicit;

use std::error::Error;
use std::fmt;

use faer::prelude::*;

use crate::ode_traits::IntegrateSys;

pub use exponential::{
    DenseExpmMethod, ExponentialEvaluator, ExponentialIntegratorBuilder, ExponentialMethod,
    KrylovOptions, LejaDdMethod, LejaOptions, LejaSpectrum, TaylorOptions,
};
pub use explicit::{ExplicitIntegratorBuilder, ExplicitMethod};
pub use implicit::{ImplicitIntegratorBuilder, ImplicitMethod};

/// A constructed, type-erased ORMATEX time integrator.
///
/// Boxed [`IntegrateSys`] trait object with time type `f64` and dense
/// state type `Mat<f64>`. Returned by the `build()` method of every
/// integrator builder.
pub type BuiltIntegrator =
    Box<dyn IntegrateSys<'static, TimeType = f64, SysStateType = Mat<f64>>>;

/// Error returned when an integrator configuration is invalid.
///
/// Produced by the `build()` methods of the integrator builders and by the
/// `FromStr` implementations of the method enums (unknown method name). The
/// human readable reason is available through `Display`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegratorBuildError {
    message: String,
}

impl IntegratorBuildError {
    /// Create an error from a message describing the invalid configuration.
    ///
    /// # Arguments
    ///
    /// * `message` - human readable description of the problem
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for IntegratorBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for IntegratorBuildError {}

/// Check that a named option is finite and strictly positive.
///
/// # Arguments
///
/// * `name` - option name used in the error message
/// * `value` - value to validate
///
/// # Errors
///
/// Returns an [`IntegratorBuildError`] if `value` is NaN, infinite, or
/// $\le 0$.
pub(crate) fn positive_f64(name: &str, value: f64) -> Result<(), IntegratorBuildError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(IntegratorBuildError::new(format!(
            "{name} must be finite and positive"
        )))
    }
}

/// Check that a named option is finite and non-negative.
///
/// # Arguments
///
/// * `name` - option name used in the error message
/// * `value` - value to validate
///
/// # Errors
///
/// Returns an [`IntegratorBuildError`] if `value` is NaN, infinite, or
/// $< 0$.
pub(crate) fn nonnegative_f64(name: &str, value: f64) -> Result<(), IntegratorBuildError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(IntegratorBuildError::new(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_common::TestLvSys;

    #[test]
    fn each_family_builds_and_steps() {
        let y0 = faer::mat![[1.0_f64], [2.0_f64]];
        let sys = TestLvSys::new();

        let mut explicit = ExplicitIntegratorBuilder::new(0.0, y0.as_ref(), ExplicitMethod::Rk4)
            .build()
            .unwrap();
        let step = explicit.step(&sys, 0.01).unwrap();
        explicit.accept_step(step);
        assert_eq!(explicit.time(), 0.01);

        let mut implicit =
            ImplicitIntegratorBuilder::new(0.0, y0.as_ref(), ImplicitMethod::Bdf1)
                .build()
                .unwrap();
        let step = implicit.step(&sys, 0.01).unwrap();
        implicit.accept_step(step);
        assert_eq!(implicit.time(), 0.01);

        let mut exponential = ExponentialIntegratorBuilder::new(
            0.0,
            y0.as_ref(),
            ExponentialMethod::Epi2,
        )
        .with_krylov(KrylovOptions::default().with_m(6).with_max_dim(20))
        .build()
        .unwrap();
        let step = exponential.step(&sys, 0.01).unwrap();
        exponential.accept_step(step);
        assert_eq!(exponential.time(), 0.01);
    }
}
