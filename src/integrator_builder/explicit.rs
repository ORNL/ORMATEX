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
//! Builder for explicit Runge-Kutta time integrators.
//!
//! Provides [`ExplicitMethod`], the set of supported explicit Runge-Kutta
//! schemes (orders 1 to 4), and [`ExplicitIntegratorBuilder`], which validates
//! the configuration and constructs a boxed [`crate::ode_rk::RkIntegrator`].
//! Explicit methods are only conditionally stable and are intended for
//! non-stiff problems or as reference solutions.
//!
//! # References
//!
//! * Hairer, E., Norsett, S. P., Wanner, G. Solving Ordinary Differential
//!   Equations I: Nonstiff Problems. Springer, 1993.
use std::str::FromStr;

use faer::prelude::*;

use crate::integrator_builder::{BuiltIntegrator, IntegratorBuildError};
use crate::ode_rk::RkIntegrator;

/// Explicit Runge-Kutta methods available from [`ExplicitIntegratorBuilder`].
///
/// Each variant is an explicit Runge-Kutta scheme of the given order with
/// as many stages as its order (family: explicit Runge-Kutta). The name
/// accepted by `FromStr` (case insensitive) is given for each variant.
///
/// # References
///
/// * Hairer, E., Norsett, S. P., Wanner, G. Solving Ordinary Differential
///   Equations I: Nonstiff Problems. Springer, 1993.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplicitMethod {
    /// Forward Euler. Order 1, 1 stage. Names: `rk1`, `forwardeuler`.
    Rk1,
    /// Explicit midpoint method. Order 2, 2 stages
    /// ($c = (0, 1/2)$, $b = (0, 1)$). Name: `rk2`.
    Rk2,
    /// Kutta's third order method. Order 3, 3 stages
    /// ($c = (0, 1/2, 1)$, $b = (1/6, 2/3, 1/6)$). Name: `rk3`.
    Rk3,
    /// Classical fourth order Runge-Kutta method. Order 4, 4 stages
    /// ($c = (0, 1/2, 1/2, 1)$, $b = (1/6, 1/3, 1/3, 1/6)$). Name: `rk4`.
    Rk4,
}

impl ExplicitMethod {
    /// Order of accuracy of the method, which is also its number of stages.
    fn order(self) -> usize {
        match self {
            Self::Rk1 => 1,
            Self::Rk2 => 2,
            Self::Rk3 => 3,
            Self::Rk4 => 4,
        }
    }
}

impl FromStr for ExplicitMethod {
    type Err = IntegratorBuildError;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method.to_ascii_lowercase().as_str() {
            "rk1" | "forwardeuler" => Ok(Self::Rk1),
            "rk2" => Ok(Self::Rk2),
            "rk3" => Ok(Self::Rk3),
            "rk4" => Ok(Self::Rk4),
            _ => Err(IntegratorBuildError::new(format!(
                "unsupported explicit time integration method: {method}"
            ))),
        }
    }
}

/// Builder for explicit Runge-Kutta integrators.
///
/// Holds the initial condition and method choice, and constructs a
/// [`BuiltIntegrator`] with [`ExplicitIntegratorBuilder::build`]. Explicit
/// methods have no tunable options.
pub struct ExplicitIntegratorBuilder {
    t0: f64,
    y0: Mat<f64>,
    method: ExplicitMethod,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_aliases() {
        assert_eq!(ExplicitMethod::from_str("forwardeuler"), Ok(ExplicitMethod::Rk1));
        assert_eq!(ExplicitMethod::from_str("RK4"), Ok(ExplicitMethod::Rk4));
    }
}

impl ExplicitIntegratorBuilder {
    /// Create a builder for an explicit Runge-Kutta integrator.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state, an $n \times 1$ column; it is copied
    /// * `method` - the explicit Runge-Kutta method to use
    pub fn new(t0: f64, y0: MatRef<'_, f64>, method: ExplicitMethod) -> Self {
        Self {
            t0,
            y0: y0.to_owned(),
            method,
        }
    }

    /// Construct the integrator.
    ///
    /// # Returns
    ///
    /// A boxed [`crate::ode_rk::RkIntegrator`] of the requested order, starting
    /// at `t0` with state `y0`.
    ///
    /// # Errors
    ///
    /// Currently no validation is performed, so this always returns `Ok`.
    /// The `Result` return type is kept for consistency with the other
    /// integrator builders.
    pub fn build(&self) -> Result<BuiltIntegrator, IntegratorBuildError> {
        Ok(Box::new(RkIntegrator::new(
            self.t0,
            self.y0.as_ref(),
            self.method.order(),
        )))
    }
}
