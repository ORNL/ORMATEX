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
//! Builder for implicit (BDF and DIRK/SDIRK) time integrators.
//!
//! Provides [`ImplicitMethod`], the supported implicit schemes, and
//! [`ImplicitIntegratorBuilder`], which validates the linear and nonlinear
//! solver tolerances and constructs a boxed [`crate::ode_implicit::BdfIntegrator`] or
//! [`crate::ode_implicit::DirkIntegrator`]. Implicit stage equations are solved
//! with a Newton-Krylov iteration, so these methods are suited to stiff
//! problems.
//!
//! # References
//!
//! * Hairer, E., Wanner, G. Solving Ordinary Differential Equations II: Stiff
//!   and Differential-Algebraic Problems. Springer, 1996.
//! * Alexander, R. Diagonally implicit Runge-Kutta methods for stiff ODEs.
//!   SIAM J. Numer. Anal. 14(6) (1977) 1006-1021.
use std::str::FromStr;

use faer::prelude::*;

use crate::ode_implicit::{BdfIntegrator, DirkIntegrator};
use crate::integrator_builder::{positive_f64, BuiltIntegrator, IntegratorBuildError};
use crate::tableau_implicit::ImplicitBT;

/// Implicit methods available from [`ImplicitIntegratorBuilder`].
///
/// The BDF variants are linear multistep methods; the remaining variants are
/// diagonally implicit Runge-Kutta (DIRK) methods defined by a Butcher tableau
/// from [`crate::tableau_implicit::ImplicitBT`]. Orders and stage counts
/// follow the tableau documentation. The name accepted by `FromStr` (case
/// insensitive) is given for each variant.
///
/// # References
///
/// * Hairer, E., Wanner, G. Solving Ordinary Differential Equations II: Stiff
///   and Differential-Algebraic Problems. Springer, 1996.
/// * Alexander, R. Diagonally implicit Runge-Kutta methods for stiff ODEs.
///   SIAM J. Numer. Anal. 14(6) (1977) 1006-1021.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplicitMethod {
    /// Backward (implicit) Euler. BDF family, order 1, 1 implicit stage.
    /// Names: `bdf1`, `backeuler`, `implicit_euler`.
    Bdf1,
    /// Two-step backward differentiation formula. BDF family, order 2. The
    /// first step, when only one past state is available, is taken with
    /// backward Euler. Name: `bdf2`.
    Bdf2,
    /// Crank-Nicolson (trapezoidal rule). ESDIRK family, order 2, 2 stages
    /// (the first stage is explicit). Name: `cn`.
    CrankNicolson,
    /// Two-stage SDIRK, $\gamma = 1 - 1/\sqrt{2}$. SDIRK family, order 2,
    /// 2 stages. Name: `sdirk22`.
    Sdirk22,
    /// Three-stage SDIRK with $\gamma = 1/4$. SDIRK family, order 2,
    /// 3 stages. Name: `sdirk32`.
    Sdirk32,
    /// Three-stage SDIRK Norsett variant with $\gamma = (3 - \sqrt{3})/6$.
    /// SDIRK family, order 2, 3 stages. Name: `sdirk32_norsett`.
    Sdirk32Norsett,
    /// Three-stage, third order SDIRK of Alexander (1977) with
    /// $\gamma \approx 0.4358665$. SDIRK family, order 3, 3 stages.
    /// Name: `sdirk33`.
    Sdirk33,
}

impl FromStr for ImplicitMethod {
    type Err = IntegratorBuildError;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method.to_ascii_lowercase().as_str() {
            "bdf1" | "backeuler" | "implicit_euler" => Ok(Self::Bdf1),
            "bdf2" => Ok(Self::Bdf2),
            "cn" => Ok(Self::CrankNicolson),
            "sdirk22" => Ok(Self::Sdirk22),
            "sdirk32" => Ok(Self::Sdirk32),
            "sdirk32_norsett" => Ok(Self::Sdirk32Norsett),
            "sdirk33" => Ok(Self::Sdirk33),
            _ => Err(IntegratorBuildError::new(format!(
                "unsupported implicit time integration method: {method}"
            ))),
        }
    }
}

/// Builder for implicit (BDF and DIRK/SDIRK) integrators.
///
/// Holds the initial condition, method choice, and Newton-Krylov solver
/// tolerances. Construct the integrator with
/// [`ImplicitIntegratorBuilder::build`].
pub struct ImplicitIntegratorBuilder {
    t0: f64,
    y0: Mat<f64>,
    method: ImplicitMethod,
    tol_lin: f64,
    tol_nlin: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_implicit_aliases() {
        assert_eq!(ImplicitMethod::from_str("backeuler"), Ok(ImplicitMethod::Bdf1));
        assert_eq!(ImplicitMethod::from_str("implicit_euler"), Ok(ImplicitMethod::Bdf1));
        assert_eq!(ImplicitMethod::from_str("CN"), Ok(ImplicitMethod::CrankNicolson));
    }
}

impl ImplicitIntegratorBuilder {
    /// Create a builder for an implicit integrator with default tolerances.
    ///
    /// The default linear and nonlinear solver tolerances are both `1e-8`.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state, an $n \times 1$ column; it is copied
    /// * `method` - the implicit method to use
    pub fn new(t0: f64, y0: MatRef<'_, f64>, method: ImplicitMethod) -> Self {
        Self {
            t0,
            y0: y0.to_owned(),
            method,
            tol_lin: 1e-8,
            tol_nlin: 1e-8,
        }
    }

    /// Set the tolerance of the linear (Krylov) solves inside the Newton iteration.
    ///
    /// Default: `1e-8`. Valid range: finite and strictly positive; this is
    /// checked by [`ImplicitIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `tol_lin` - linear solver tolerance
    pub fn with_tol_lin(mut self, tol_lin: f64) -> Self {
        self.tol_lin = tol_lin;
        self
    }

    /// Set the tolerance of the Newton iteration for the implicit stage equations.
    ///
    /// Default: `1e-8`. Valid range: finite and strictly positive; this is
    /// checked by [`ImplicitIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `tol_nlin` - nonlinear (Newton) solver tolerance
    pub fn with_tol_nlin(mut self, tol_nlin: f64) -> Self {
        self.tol_nlin = tol_nlin;
        self
    }

    /// Construct the integrator.
    ///
    /// # Returns
    ///
    /// A boxed [`crate::ode_implicit::BdfIntegrator`] for `Bdf1` and `Bdf2`,
    /// or a boxed [`crate::ode_implicit::DirkIntegrator`] with the matching
    /// Butcher tableau for the other methods.
    ///
    /// # Errors
    ///
    /// Returns an [`IntegratorBuildError`] if
    ///
    /// * `tol_lin` is not finite and strictly positive, or
    /// * `tol_nlin` is not finite and strictly positive.
    pub fn build(&self) -> Result<BuiltIntegrator, IntegratorBuildError> {
        positive_f64("tol_lin", self.tol_lin)?;
        positive_f64("tol_nlin", self.tol_nlin)?;

        match self.method {
            ImplicitMethod::Bdf1 | ImplicitMethod::Bdf2 => Ok(Box::new(BdfIntegrator::new(
                self.t0,
                self.y0.as_ref(),
                match self.method {
                    ImplicitMethod::Bdf1 => 1,
                    ImplicitMethod::Bdf2 => 2,
                    _ => unreachable!(),
                },
                self.tol_lin,
                self.tol_nlin,
            ))),
            ImplicitMethod::CrankNicolson => self.build_dirk(ImplicitBT::crank_nicolson()),
            ImplicitMethod::Sdirk22 => self.build_dirk(ImplicitBT::sdirk22()),
            ImplicitMethod::Sdirk32 => self.build_dirk(ImplicitBT::sdirk32()),
            ImplicitMethod::Sdirk32Norsett => self.build_dirk(ImplicitBT::sdirk32_norsett()),
            ImplicitMethod::Sdirk33 => self.build_dirk(ImplicitBT::sdirk33()),
        }
    }

    /// Construct a DIRK integrator from a Butcher tableau.
    fn build_dirk(&self, tableau: ImplicitBT) -> Result<BuiltIntegrator, IntegratorBuildError> {
        Ok(Box::new(DirkIntegrator::new(
            self.t0,
            self.y0.as_ref(),
            tableau,
            self.tol_lin,
            self.tol_nlin,
        )))
    }
}
