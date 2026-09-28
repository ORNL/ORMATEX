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
use std::str::FromStr;

use faer::prelude::*;

use crate::integrator_builder::{BuiltIntegrator, IntegratorBuildError};
use crate::ode_rk::RkIntegrator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplicitMethod {
    Rk1,
    Rk2,
    Rk3,
    Rk4,
}

impl ExplicitMethod {
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
    pub fn new(t0: f64, y0: MatRef<'_, f64>, method: ExplicitMethod) -> Self {
        Self {
            t0,
            y0: y0.to_owned(),
            method,
        }
    }

    pub fn build(&self) -> Result<BuiltIntegrator, IntegratorBuildError> {
        Ok(Box::new(RkIntegrator::new(
            self.t0,
            self.y0.as_ref(),
            self.method.order(),
        )))
    }
}
