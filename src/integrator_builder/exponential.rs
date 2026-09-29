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

use crate::matexp_cauchy;
use crate::matexp_krylov::KrylovExpm;
use crate::matexp_leja::{
    LejaEllipseAdapterArnoldiIOM, LejaEllipseAdapterStatic, LejaPhiEval, LejaPoints,
};
use crate::matexp_pade::PadeExpm;
use crate::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use crate::ode_epirk::EpirkIntegrator;
use crate::ode_exprb::ExprbIntegrator;

use crate::integrator_builder::{
    nonnegative_f64, positive_f64, BuiltIntegrator, IntegratorBuildError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExponentialMethod {
    Epi2,
    Exprb2,
    Epi3,
    Exprb3,
}

impl FromStr for ExponentialMethod {
    type Err = IntegratorBuildError;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method.to_ascii_lowercase().as_str() {
            "epi2" => Ok(Self::Epi2),
            "exprb2" => Ok(Self::Exprb2),
            "epi3" => Ok(Self::Epi3),
            "exprb3" => Ok(Self::Exprb3),
            _ => Err(IntegratorBuildError::new(format!(
                "unsupported exponential time integration method: {method}"
            ))),
        }
    }
}

impl ExponentialMethod {
    fn name(self) -> &'static str {
        match self {
            Self::Epi2 => "epi2",
            Self::Exprb2 => "exprb2",
            Self::Epi3 => "epi3",
            Self::Exprb3 => "exprb3",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseExpmMethod {
    Pade,
    Cram16,
    Parabolic,
}

impl Default for DenseExpmMethod {
    fn default() -> Self {
        Self::Pade
    }
}

impl FromStr for DenseExpmMethod {
    type Err = IntegratorBuildError;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method.to_ascii_lowercase().as_str() {
            "pade" => Ok(Self::Pade),
            "cram" | "cram_16" => Ok(Self::Cram16),
            "parabolic" => Ok(Self::Parabolic),
            _ => Err(IntegratorBuildError::new(format!(
                "unsupported dense exponential method: {method}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct KrylovOptions {
    dense_method: DenseExpmMethod,
    m: usize,
    max_dim: usize,
    iom: usize,
    tol: f64,
}

impl Default for KrylovOptions {
    fn default() -> Self {
        Self {
            dense_method: DenseExpmMethod::Pade,
            m: 100,
            max_dim: 100,
            iom: 2,
            tol: 1e-8,
        }
    }
}

impl KrylovOptions {
    pub fn with_dense_method(mut self, dense_method: DenseExpmMethod) -> Self {
        self.dense_method = dense_method;
        self
    }

    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn with_max_dim(mut self, max_dim: usize) -> Self {
        self.max_dim = max_dim;
        self
    }

    pub fn with_iom(mut self, iom: usize) -> Self {
        self.iom = iom;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn validate(&self) -> Result<(), IntegratorBuildError> {
        positive_f64("Krylov tolerance", self.tol)?;
        if self.max_dim == 0 {
            return Err(IntegratorBuildError::new(
                "Krylov max_dim must be positive",
            ));
        }
        if self.m == 0 || self.m > self.max_dim {
            return Err(IntegratorBuildError::new(
                "Krylov m must be positive and no greater than max_dim",
            ));
        }
        Ok(())
    }

    fn evaluator(&self) -> KrylovExpm {
        let dense: Box<dyn DensePhikvEvaluator> = match self.dense_method {
            DenseExpmMethod::Pade => Box::new(PadeExpm::new(12)),
            DenseExpmMethod::Cram16 => Box::new(matexp_cauchy::gen_cram_expm(16)),
            DenseExpmMethod::Parabolic => Box::new(matexp_cauchy::gen_parabolic_expm(24)),
        };
        KrylovExpm::new(
            dense,
            self.m.min(50),
            self.max_dim,
            self.tol,
            Some(self.iom),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LejaDdMethod {
    Phi,
    Taylor,
}

impl Default for LejaDdMethod {
    fn default() -> Self {
        Self::Phi
    }
}

impl FromStr for LejaDdMethod {
    type Err = IntegratorBuildError;

    fn from_str(method: &str) -> Result<Self, Self::Err> {
        match method.to_ascii_lowercase().as_str() {
            "dd_phi" => Ok(Self::Phi),
            "dd_taylor" => Ok(Self::Taylor),
            _ => Err(IntegratorBuildError::new(format!(
                "unsupported Leja divided-difference method: {method}"
            ))),
        }
    }
}

impl LejaDdMethod {
    fn name(self) -> &'static str {
        match self {
            Self::Phi => "dd_phi",
            Self::Taylor => "dd_taylor",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LejaSpectrum {
    Static { a: f64, b: f64, c: f64 },
    Adaptive {
        a: f64,
        b: f64,
        c: f64,
        spec_tol: f64,
        spec_iter: usize,
        iom: usize,
        safety_factor: f64,
    },
}

impl Default for LejaSpectrum {
    fn default() -> Self {
        Self::Adaptive {
            a: -1.0,
            b: 0.0,
            c: 1.0,
            spec_tol: 1.0e-8,
            spec_iter: 20,
            iom: 2,
            safety_factor: 1.05,
        }
    }
}

impl LejaSpectrum {
    pub fn static_bounds(a: f64, b: f64, c: f64) -> Self {
        Self::Static { a, b, c }
    }

    pub fn adaptive(
        a: f64,
        b: f64,
        c: f64,
        spec_tol: f64,
        spec_iter: usize,
        iom: usize,
        safety_factor: f64,
    ) -> Self {
        Self::Adaptive {
            a,
            b,
            c,
            spec_tol,
            spec_iter,
            iom,
            safety_factor,
        }
    }

    fn bounds(&self) -> (f64, f64, f64) {
        match self {
            Self::Static { a, b, c } | Self::Adaptive { a, b, c, .. } => (*a, *b, *c),
        }
    }

    fn validate(&self) -> Result<(), IntegratorBuildError> {
        let (a, b, c) = self.bounds();
        if !a.is_finite() || !b.is_finite() || !c.is_finite() || a > b || c < 0.0 {
            return Err(IntegratorBuildError::new(
                "invalid Leja spectrum bounds",
            ));
        }
        if let Self::Adaptive {
            spec_tol,
            safety_factor,
            ..
        } = self
        {
            nonnegative_f64("Leja spectrum tolerance", *spec_tol)?;
            positive_f64("Leja spectrum safety factor", *safety_factor)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LejaOptions {
    m: usize,
    max_substeps: usize,
    tol: f64,
    dd_method: LejaDdMethod,
    krylov_reuse: bool,
    spectrum: LejaSpectrum,
}

impl Default for LejaOptions {
    fn default() -> Self {
        Self {
            m: 100,
            max_substeps: 0,
            tol: 1e-8,
            dd_method: LejaDdMethod::default(),
            krylov_reuse: false,
            spectrum: LejaSpectrum::default(),
        }
    }
}

impl LejaOptions {
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
        self.max_substeps = max_substeps;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn with_dd_method(mut self, dd_method: LejaDdMethod) -> Self {
        self.dd_method = dd_method;
        self
    }

    pub fn with_krylov_reuse(mut self, krylov_reuse: bool) -> Self {
        self.krylov_reuse = krylov_reuse;
        self
    }

    pub fn with_spectrum(mut self, spectrum: LejaSpectrum) -> Self {
        self.spectrum = spectrum;
        self
    }

    fn validate(&self) -> Result<(), IntegratorBuildError> {
        positive_f64("Leja tolerance", self.tol)?;
        if self.m == 0 {
            return Err(IntegratorBuildError::new("Leja m must be positive"));
        }
        self.m.checked_add(2).ok_or_else(|| {
            IntegratorBuildError::new("Leja m is too large")
        })?;
        self.spectrum.validate()
    }

    fn evaluator(&self) -> Result<LejaPhiEval, IntegratorBuildError> {
        self.validate()?;
        let points = LejaPoints::new_from_fn("leja_circle").slice(0, self.m + 2);
        let (a, b, c) = self.spectrum.bounds();
        let adapter: Box<dyn crate::matexp_leja::GetSpectrumBounds> = match &self.spectrum {
            LejaSpectrum::Static { .. } => Box::new(LejaEllipseAdapterStatic::new(a, b, c)),
            LejaSpectrum::Adaptive {
                spec_tol,
                spec_iter,
                iom,
                safety_factor,
                ..
            } => Box::new(LejaEllipseAdapterArnoldiIOM::new(
                a,
                b,
                c,
                *spec_tol,
                *spec_iter,
                *iom,
                *safety_factor,
            )),
        };
        let mut evaluator = LejaPhiEval::new(
            points,
            self.m.min(800),
            self.tol,
            "clapm",
            self.dd_method.name(),
            self.krylov_reuse,
            adapter,
        );
        evaluator.set_max_substeps(self.max_substeps);
        Ok(evaluator)
    }
}

#[derive(Clone, Debug)]
pub struct TaylorOptions {
    m: usize,
    tol: f64,
    dd_method: LejaDdMethod,
    krylov_reuse: bool,
    a: f64,
    b: f64,
    c: f64,
}

impl Default for TaylorOptions {
    fn default() -> Self {
        Self {
            m: 100,
            tol: 1e-8,
            dd_method: LejaDdMethod::default(),
            krylov_reuse: false,
            a: -1.0,
            b: 0.0,
            c: 1.0,
        }
    }
}

impl TaylorOptions {
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn with_dd_method(mut self, dd_method: LejaDdMethod) -> Self {
        self.dd_method = dd_method;
        self
    }

    pub fn with_krylov_reuse(mut self, krylov_reuse: bool) -> Self {
        self.krylov_reuse = krylov_reuse;
        self
    }

    pub fn with_bounds(mut self, a: f64, b: f64, c: f64) -> Self {
        self.a = a;
        self.b = b;
        self.c = c;
        self
    }

    fn evaluator(&self) -> Result<LejaPhiEval, IntegratorBuildError> {
        positive_f64("Taylor tolerance", self.tol)?;
        if self.m == 0 {
            return Err(IntegratorBuildError::new("Taylor m must be positive"));
        }
        if !self.a.is_finite()
            || !self.b.is_finite()
            || !self.c.is_finite()
            || self.a > self.b
            || self.c < 0.0
        {
            return Err(IntegratorBuildError::new(
                "invalid Taylor spectrum bounds",
            ));
        }
        let points = LejaPoints::new(vec![0.0; self.m], vec![0.0; self.m]);
        let adapter = LejaEllipseAdapterStatic::new(self.a, self.b, self.c);
        Ok(LejaPhiEval::new(
            points,
            self.m.min(800),
            self.tol,
            "taylor",
            self.dd_method.name(),
            self.krylov_reuse,
            Box::new(adapter),
        ))
    }
}

#[derive(Clone, Debug)]
pub enum ExponentialEvaluator {
    Krylov(KrylovOptions),
    Leja(LejaOptions),
    Taylor(TaylorOptions),
}

impl Default for ExponentialEvaluator {
    fn default() -> Self {
        Self::Krylov(KrylovOptions::default())
    }
}

pub struct ExponentialIntegratorBuilder {
    t0: f64,
    y0: Mat<f64>,
    method: ExponentialMethod,
    tol_fdt: f64,
    evaluator: ExponentialEvaluator,
}

impl ExponentialIntegratorBuilder {
    pub fn new(t0: f64, y0: MatRef<'_, f64>, method: ExponentialMethod) -> Self {
        Self {
            t0,
            y0: y0.to_owned(),
            method,
            tol_fdt: 1e-8,
            evaluator: ExponentialEvaluator::default(),
        }
    }

    pub fn with_tol_fdt(mut self, tol_fdt: f64) -> Self {
        self.tol_fdt = tol_fdt;
        self
    }

    pub fn with_evaluator(mut self, evaluator: ExponentialEvaluator) -> Self {
        self.evaluator = evaluator;
        self
    }

    pub fn with_krylov(self, options: KrylovOptions) -> Self {
        self.with_evaluator(ExponentialEvaluator::Krylov(options))
    }

    pub fn with_leja(self, options: LejaOptions) -> Self {
        self.with_evaluator(ExponentialEvaluator::Leja(options))
    }

    pub fn with_taylor(self, options: TaylorOptions) -> Self {
        self.with_evaluator(ExponentialEvaluator::Taylor(options))
    }

    pub fn build(&self) -> Result<BuiltIntegrator, IntegratorBuildError> {
        if !self.tol_fdt.is_finite() {
            return Err(IntegratorBuildError::new("tol_fdt must be finite"));
        }
        match &self.evaluator {
            ExponentialEvaluator::Krylov(options) => {
                options.validate()?;
                self.build_with(options.evaluator())
            }
            ExponentialEvaluator::Leja(options) => self.build_with(options.evaluator()?),
            ExponentialEvaluator::Taylor(options) => self.build_with(options.evaluator()?),
        }
    }

    fn build_with<T>(&self, evaluator: T) -> Result<BuiltIntegrator, IntegratorBuildError>
    where
        T: LinOpPhikvEvaluator + 'static,
    {
        let method = self.method.name().to_string();
        let solver: BuiltIntegrator = match self.method {
            ExponentialMethod::Exprb3 => Box::new(
                ExprbIntegrator::new(self.t0, self.y0.as_ref(), method, evaluator)
                    .with_opt(String::from("tol_fdt"), self.tol_fdt),
            ),
            ExponentialMethod::Epi2
            | ExponentialMethod::Exprb2
            | ExponentialMethod::Epi3 => Box::new(
                EpirkIntegrator::new(self.t0, self.y0.as_ref(), method, evaluator)
                    .with_opt(String::from("tol_fdt"), self.tol_fdt),
            ),
        };
        Ok(solver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_common::TestLvSys;

    #[test]
    fn builds_each_method_and_evaluator() {
        let y0 = faer::mat![[1.0_f64], [2.0_f64]];
        let sys = TestLvSys::new();
        let methods = [
            ExponentialMethod::Epi2,
            ExponentialMethod::Exprb2,
            ExponentialMethod::Epi3,
            ExponentialMethod::Exprb3,
        ];

        for method in methods {
            let mut solver = ExponentialIntegratorBuilder::new(0.0, y0.as_ref(), method)
                .with_krylov(KrylovOptions::default().with_m(6).with_max_dim(20))
                .build()
                .unwrap();
            let step = solver.step(&sys, 0.01).unwrap();
            solver.accept_step(step);
            assert_eq!(solver.time(), 0.01);
        }

        for evaluator in [
            ExponentialEvaluator::Krylov(
                KrylovOptions::default().with_m(6).with_max_dim(20),
            ),
            ExponentialEvaluator::Leja(
                LejaOptions::default().with_m(6).with_spectrum(LejaSpectrum::static_bounds(
                    -10.0, 0.0, 0.0,
                )),
            ),
            ExponentialEvaluator::Taylor(TaylorOptions::default().with_m(6)),
        ] {
            ExponentialIntegratorBuilder::new(0.0, y0.as_ref(), ExponentialMethod::Epi2)
                .with_evaluator(evaluator)
                .build()
                .unwrap();
        }
    }

    #[test]
    fn rejects_invalid_options() {
        let y0 = faer::mat![[1.0_f64]];
        assert!(ExponentialIntegratorBuilder::new(
            0.0,
            y0.as_ref(),
            ExponentialMethod::Epi2,
        )
        .with_krylov(KrylovOptions::default().with_m(21).with_max_dim(20))
        .build()
        .is_err());
    }

    #[test]
    fn parses_exponential_and_evaluator_names() {
        assert_eq!(ExponentialMethod::from_str("exprb3"), Ok(ExponentialMethod::Exprb3));
        assert_eq!(DenseExpmMethod::from_str("cram"), Ok(DenseExpmMethod::Cram16));
        assert_eq!(LejaDdMethod::from_str("dd_taylor"), Ok(LejaDdMethod::Taylor));
    }
}
