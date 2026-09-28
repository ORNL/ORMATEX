use std::str::FromStr;

use faer::prelude::*;

use crate::ode_implicit::{BdfIntegrator, DirkIntegrator};
use crate::integrator_builder::{positive_f64, BuiltIntegrator, IntegratorBuildError};
use crate::tableau_implicit::ImplicitBT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplicitMethod {
    Bdf1,
    Bdf2,
    CrankNicolson,
    Sdirk22,
    Sdirk32,
    Sdirk32Norsett,
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
    pub fn new(t0: f64, y0: MatRef<'_, f64>, method: ImplicitMethod) -> Self {
        Self {
            t0,
            y0: y0.to_owned(),
            method,
            tol_lin: 1e-8,
            tol_nlin: 1e-8,
        }
    }

    pub fn with_tol_lin(mut self, tol_lin: f64) -> Self {
        self.tol_lin = tol_lin;
        self
    }

    pub fn with_tol_nlin(mut self, tol_nlin: f64) -> Self {
        self.tol_nlin = tol_nlin;
        self
    }

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
