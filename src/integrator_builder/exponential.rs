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
//! Builder for exponential time integrators and their phi-function evaluators.
//!
//! Provides [`ExponentialMethod`] (EPI and exponential Rosenbrock schemes),
//! the evaluator options [`KrylovOptions`], [`LejaOptions`] and
//! [`TaylorOptions`] (selected through [`ExponentialEvaluator`]), and
//! [`ExponentialIntegratorBuilder`], which validates the configuration and
//! constructs a boxed [`crate::ode_epirk::EpirkIntegrator`] or
//! [`crate::ode_exprb::ExprbIntegrator`].
//!
//! Exponential integrators advance $y^\prime  = f(t, y)$ by linearizing about the
//! current state with Jacobian $J$ and evaluating products of
//! $\exp(J h)$ and $\varphi_k(J h)$ with vectors, where
//! $\varphi_0(z) = e^z$ and $\varphi_{k+1}(z) = (\varphi_k(z) - 1/k!)/z$. The
//! evaluator options control how those products are approximated.
//!
//! # References
//!
//! * Hochbruck, M., Ostermann, A. Exponential integrators. Acta Numerica 19
//!   (2010) 209-286. doi:10.1017/S0962492910000048
//! * Tokman, M. Efficient integration of large stiff systems of ODEs with
//!   exponential propagation iterative (EPI) methods. J. Comput. Phys. 213
//!   (2006) 748-776. doi:10.1016/j.jcp.2005.08.032
//! * Hochbruck, M., Ostermann, A., Schweitzer, J. Exponential Rosenbrock-type
//!   methods. SIAM J. Numer. Anal. 47(1) (2009) 786-803.
//!   doi:10.1137/080717717
//! * Gaudreault, S., Pudykiewicz, J. A. An efficient exponential time
//!   integration method for the numerical solution of the shallow water
//!   equations on the sphere. J. Comput. Phys. 322 (2016) 827-848.
//! * Saad, Y. Analysis of some Krylov subspace approximations to the matrix
//!   exponential operator. SIAM J. Numer. Anal. 29(1) (1992) 209-228.
//! * Caliari, M., Vianello, M., Bergamaschi, L. Interpolating discrete
//!   advection-diffusion propagators at Leja sequences. J. Comput. Appl. Math.
//!   172(1) (2004) 79-99.
//! * Caliari, M., Cassini, F., Zivcovich, F. BAMPHI: matrix-free and
//!   transpose-free action of linear combinations of phi-functions from
//!   exponential integrators. J. Comput. Appl. Math. 423 (2023) 114973.
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

/// Exponential time integration methods available from
/// [`ExponentialIntegratorBuilder`].
///
/// The name accepted by `FromStr` (case insensitive) is the lower case variant
/// name, for example `epi2`.
///
/// # References
///
/// * Tokman, M. Efficient integration of large stiff systems of ODEs with
///   exponential propagation iterative (EPI) methods. J. Comput. Phys. 213
///   (2006) 748-776. doi:10.1016/j.jcp.2005.08.032
/// * Hochbruck, M., Ostermann, A., Schweitzer, J. Exponential Rosenbrock-type
///   methods. SIAM J. Numer. Anal. 47(1) (2009) 786-803.
///   doi:10.1137/080717717
/// * Gaudreault, S., Pudykiewicz, J. A. J. Comput. Phys. 322 (2016) 827-848.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExponentialMethod {
    /// EPI2, order 2 (family: exponential propagation iterative, EPI).
    ///
    /// Single stage, one-step method: $y_{n+1} = y_n + h \varphi_1(h J_n) f(y_n)$.
    Epi2,
    /// Exponential Rosenbrock-Euler, order 2 (family: exponential Rosenbrock).
    ///
    /// Single stage. In this crate it uses the same update, and the same
    /// integrator type, as `Epi2`.
    Exprb2,
    /// EPI3, order 3 (family: EPI).
    ///
    /// Two-step method that uses the previous accepted state to form the
    /// nonlinear remainder. The first step, when no previous state exists, is
    /// taken with the order 2 update.
    Epi3,
    /// Exponential Rosenbrock method of order 3 with an embedded order 2
    /// error estimate (exprb32; family: exponential Rosenbrock).
    ///
    /// Two stages, one-step method. The step result carries an error
    /// estimate from the difference of the two stage solutions.
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
    /// Lower case method name expected by the integrator constructors.
    fn name(self) -> &'static str {
        match self {
            Self::Epi2 => "epi2",
            Self::Exprb2 => "exprb2",
            Self::Epi3 => "epi3",
            Self::Exprb3 => "exprb3",
        }
    }
}

/// Dense matrix exponential and phi-function method used inside the Krylov
/// subspace by [`KrylovOptions`].
///
/// The Krylov method reduces the problem to a small dense Hessenberg matrix
/// $H$ and evaluates $\varphi_k(H)$ with one of these methods. Default:
/// `Pade`.
///
/// # References
///
/// * Higham, N. J. The scaling and squaring method for the matrix exponential
///   revisited. SIAM J. Matrix Anal. Appl. 26(4) (2005) 1179-1193.
///   doi:10.1137/04061101X
/// * Pusa, M. Rational approximations to the matrix exponential in burnup
///   calculations. Nucl. Sci. Eng. 169(2) (2011) 155-167.
///   doi:10.13182/NSE10-81
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseExpmMethod {
    /// Pade approximation with scaling and squaring (up to 12 squarings).
    /// Name: `pade`.
    Pade,
    /// Contour integral rational approximation, CRAM of order 16
    /// (Chebyshev rational approximation method). Names: `cram`, `cram_16`.
    Cram16,
    /// Contour integral rational approximation on a parabolic contour of
    /// order 24. Name: `parabolic`.
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

/// Options for the Krylov subspace phi-function evaluator.
///
/// The action of $\varphi_k(hA)$ on a vector is approximated in the Krylov
/// subspace $\mathcal K_m(A, v)$ built by Arnoldi iteration. The default
/// values (see the `Default` impl) are: dense method `Pade`, `m = 100`,
/// `max_dim = 100`, `iom = 2`, `tol = 1e-8`.
///
/// Options are validated when the integrator is built; see
/// [`ExponentialIntegratorBuilder::build`].
///
/// # References
///
/// * Saad, Y. Analysis of some Krylov subspace approximations to the matrix
///   exponential operator. SIAM J. Numer. Anal. 29(1) (1992) 209-228.
/// * Caliari, M., Cassini, F., Zivcovich, F. BAMPHI: matrix-free and
///   transpose-free action of linear combinations of phi-functions from
///   exponential integrators. J. Comput. Appl. Math. 423 (2023) 114973.
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
    /// Set the dense matrix exponential method used on the projected matrix.
    ///
    /// Default: [`DenseExpmMethod::Pade`].
    ///
    /// # Arguments
    ///
    /// * `dense_method` - dense exponential and phi-function method
    pub fn with_dense_method(mut self, dense_method: DenseExpmMethod) -> Self {
        self.dense_method = dense_method;
        self
    }

    /// Set the initial Krylov subspace dimension.
    ///
    /// Default: `100`. Valid range: $1 \le m \le$ `max_dim`, checked by
    /// [`ExponentialIntegratorBuilder::build`]. The subspace dimension may be
    /// increased adaptively up to `max_dim` if the tolerance is not met. Note
    /// that the evaluator is currently created with an initial dimension of
    /// `min(m, 50)`, so values of `m` above 50 have no additional effect on the
    /// starting dimension.
    ///
    /// # Arguments
    ///
    /// * `m` - initial Krylov subspace dimension
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the maximum Krylov subspace dimension.
    ///
    /// Default: `100`. Valid range: `max_dim` $\ge 1$ and `max_dim` $\ge$ `m`,
    /// checked by [`ExponentialIntegratorBuilder::build`]. This also sets the
    /// size of the preallocated Arnoldi storage.
    ///
    /// # Arguments
    ///
    /// * `max_dim` - maximum Krylov subspace dimension
    pub fn with_max_dim(mut self, max_dim: usize) -> Self {
        self.max_dim = max_dim;
        self
    }

    /// Set the incomplete orthogonalization depth of the Arnoldi iteration.
    ///
    /// Each new basis vector is orthogonalized against at most the previous
    /// `iom` basis vectors. Default: `2`. This value is not range checked by
    /// [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `iom` - incomplete orthogonalization depth
    pub fn with_iom(mut self, iom: usize) -> Self {
        self.iom = iom;
        self
    }

    /// Set the convergence tolerance of the Krylov approximation.
    ///
    /// Default: `1e-8`. Valid range: finite and strictly positive, checked by
    /// [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `tol` - Krylov error tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Check the Krylov options.
    ///
    /// # Errors
    ///
    /// Returns an error if `tol` is not finite and positive, `max_dim` is 0,
    /// `m` is 0, or `m` is greater than `max_dim`.
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

/// Method used to compute the divided differences (Newton polynomial
/// coefficients) for Leja and Taylor phi-function evaluators.
///
/// Default: `Phi`. The name accepted by `FromStr` is given for each variant.
///
/// # References
///
/// * Zivcovich, F. Fast and accurate computation of divided differences for
///   analytic functions, with an application to the exponential function.
///   Dolomites Res. Notes Approx. 12 (2019).
/// * Caliari, M. Accurate evaluation of divided differences for polynomial
///   interpolation of exponential integrators. Computing 80 (2007).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LejaDdMethod {
    /// Divided differences by the `dd_phi` method of Zivcovich (2019).
    /// Name: `dd_phi`.
    Phi,
    /// Divided differences from a Taylor series method, following Caliari
    /// (2007). Name: `dd_taylor`.
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
    /// Method name string expected by the Leja evaluator.
    fn name(self) -> &'static str {
        match self {
            Self::Phi => "dd_phi",
            Self::Taylor => "dd_taylor",
        }
    }
}

/// Spectrum bounds used to scale and shift the Leja interpolation points.
///
/// The spectrum of the (scaled) Jacobian is enclosed by an ellipse described
/// by three numbers: `a` is the minimum real part of the eigenvalues, `b` is
/// the maximum real part, and `c` is the maximum magnitude of the imaginary
/// part. Valid bounds are finite with `a <= b` and `c >= 0`; this is checked
/// by [`ExponentialIntegratorBuilder::build`]. The default is an `Adaptive`
/// spectrum with `a = -1`, `b = 0`, `c = 1`, `spec_tol = 1e-8`,
/// `spec_iter = 20`, `iom = 2` and `safety_factor = 1.05`.
#[derive(Clone, Debug, PartialEq)]
pub enum LejaSpectrum {
    /// Fixed spectrum bounds that are never updated.
    Static {
        /// Minimum real part of the spectrum.
        a: f64,
        /// Maximum real part of the spectrum.
        b: f64,
        /// Maximum magnitude of the imaginary part of the spectrum.
        c: f64,
    },
    /// Spectrum bounds re-estimated from the operator by Arnoldi iteration.
    Adaptive {
        /// Initial minimum real part of the spectrum.
        a: f64,
        /// Initial maximum real part of the spectrum.
        b: f64,
        /// Initial maximum magnitude of the imaginary part of the spectrum.
        c: f64,
        /// Tolerance on the change of the operator norm estimate that triggers
        /// re-estimation of the bounds. Valid range: finite and $\ge 0$.
        spec_tol: f64,
        /// Maximum number of Arnoldi iterations used to estimate the
        /// spectrum. Not range checked by `build()`.
        spec_iter: usize,
        /// Incomplete orthogonalization depth of the Arnoldi iteration used
        /// to estimate the spectrum. Not range checked by `build()`.
        iom: usize,
        /// Factor applied to the estimated `a` and `c` bounds to enlarge the
        /// ellipse. Valid range: finite and strictly positive.
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
    /// Create fixed spectrum bounds.
    ///
    /// # Arguments
    ///
    /// * `a` - minimum real part of the spectrum
    /// * `b` - maximum real part of the spectrum, `a <= b`
    /// * `c` - maximum magnitude of the imaginary part, `c >= 0`
    pub fn static_bounds(a: f64, b: f64, c: f64) -> Self {
        Self::Static { a, b, c }
    }

    /// Create adaptive spectrum bounds estimated by Arnoldi iteration.
    ///
    /// # Arguments
    ///
    /// * `a` - initial minimum real part of the spectrum
    /// * `b` - initial maximum real part of the spectrum, `a <= b`
    /// * `c` - initial maximum magnitude of the imaginary part, `c >= 0`
    /// * `spec_tol` - tolerance that triggers re-estimation, finite and $\ge 0$
    /// * `spec_iter` - maximum number of Arnoldi iterations for the estimate
    /// * `iom` - incomplete orthogonalization depth for the estimate
    /// * `safety_factor` - factor applied to the estimated bounds, finite and $> 0$
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

    /// The `(a, b, c)` bounds of either variant.
    fn bounds(&self) -> (f64, f64, f64) {
        match self {
            Self::Static { a, b, c } | Self::Adaptive { a, b, c, .. } => (*a, *b, *c),
        }
    }

    /// Check the spectrum bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if `a`, `b` or `c` is not finite, `a > b`, or
    /// `c < 0`. For `Adaptive`, also if `spec_tol` is not finite and
    /// non-negative or `safety_factor` is not finite and positive.
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

/// Options for the Leja point phi-function evaluator.
///
/// The action of $\varphi_k(hA)$ on a vector is approximated by a Newton
/// interpolation polynomial at Leja points (complex conjugate Leja point
/// method, CLaPM), which needs only matrix-vector products. The default values
/// (see the `Default` impl) are: `m = 100`, `max_substeps = 0`, `tol = 1e-8`,
/// `dd_method = LejaDdMethod::Phi`, `krylov_reuse = false`, and the default
/// [`LejaSpectrum`].
///
/// Options are validated when the integrator is built; see
/// [`ExponentialIntegratorBuilder::build`].
///
/// # References
///
/// * Caliari, M., Vianello, M., Bergamaschi, L. Interpolating discrete
///   advection-diffusion propagators at Leja sequences. J. Comput. Appl. Math.
///   172(1) (2004) 79-99.
/// * Zivcovich, F. Fast and accurate computation of divided differences for
///   analytic functions, with an application to the exponential function.
///   Dolomites Res. Notes Approx. 12 (2019).
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
    /// Set the maximum degree of the Leja interpolation polynomial.
    ///
    /// Default: `100`. Valid range: $m \ge 1$ (and `m + 2` must not overflow),
    /// checked by [`ExponentialIntegratorBuilder::build`]. Values above 800
    /// are clamped to 800 by the evaluator.
    ///
    /// # Arguments
    ///
    /// * `m` - maximum polynomial degree
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the number of substeps used to evaluate one exponential.
    ///
    /// Default: `0`, which means no substepping (a single polynomial over the
    /// full step). A value $N > 0$ evaluates the product in $N$ equal
    /// substeps of size $h / N$. Any `usize` value is accepted.
    ///
    /// # Arguments
    ///
    /// * `max_substeps` - number of substeps, or 0 to disable substepping
    pub fn with_max_substeps(mut self, max_substeps: usize) -> Self {
        self.max_substeps = max_substeps;
        self
    }

    /// Set the convergence tolerance of the Leja polynomial.
    ///
    /// Default: `1e-8`. Valid range: finite and strictly positive, checked by
    /// [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `tol` - polynomial approximation tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the divided difference method.
    ///
    /// Default: [`LejaDdMethod::Phi`].
    ///
    /// # Arguments
    ///
    /// * `dd_method` - divided difference method
    pub fn with_dd_method(mut self, dd_method: LejaDdMethod) -> Self {
        self.dd_method = dd_method;
        self
    }

    /// Enable or disable re-use of Krylov (Arnoldi) information in the
    /// Leja polynomial evaluation.
    ///
    /// When enabled, information from the spectrum estimation is re-used,
    /// for example the Ritz values as interpolation points. Default: `false`.
    ///
    /// # Arguments
    ///
    /// * `krylov_reuse` - `true` to re-use Krylov information
    pub fn with_krylov_reuse(mut self, krylov_reuse: bool) -> Self {
        self.krylov_reuse = krylov_reuse;
        self
    }

    /// Set the spectrum bounds used to scale the Leja points.
    ///
    /// Default: `LejaSpectrum::default()` (adaptive). See [`LejaSpectrum`] for
    /// the valid ranges, which are checked by
    /// [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `spectrum` - static or adaptive spectrum bounds
    pub fn with_spectrum(mut self, spectrum: LejaSpectrum) -> Self {
        self.spectrum = spectrum;
        self
    }

    /// Check the Leja options.
    ///
    /// # Errors
    ///
    /// Returns an error if `tol` is not finite and positive, `m` is 0, `m + 2`
    /// overflows, or the spectrum bounds are invalid.
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

    /// Validate the options and construct the Leja evaluator.
    ///
    /// # Errors
    ///
    /// Returns an error if `LejaOptions::validate` fails.
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

/// Options for the truncated Taylor series phi-function evaluator.
///
/// The action of $\varphi_k(hA)$ on a vector is approximated by a Taylor
/// polynomial, using a static spectrum ellipse to shift and scale the
/// operator. This is cheap but only accurate for small $\Vert hA\Vert $. The default
/// values (see the `Default` impl) are: `m = 100`, `tol = 1e-8`,
/// `dd_method = LejaDdMethod::Phi`, `krylov_reuse = false`, and bounds
/// `a = -1`, `b = 0`, `c = 1`.
///
/// Options are validated when the integrator is built; see
/// [`ExponentialIntegratorBuilder::build`].
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
    /// Set the maximum number of Taylor series terms.
    ///
    /// Default: `100`. Valid range: $m \ge 1$, checked by
    /// [`ExponentialIntegratorBuilder::build`]. Values above 800 are clamped
    /// to 800 by the evaluator.
    ///
    /// # Arguments
    ///
    /// * `m` - maximum number of terms
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set the convergence tolerance of the Taylor series.
    ///
    /// Default: `1e-8`. Valid range: finite and strictly positive, checked by
    /// [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `tol` - series truncation tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the divided difference method passed to the evaluator.
    ///
    /// Default: [`LejaDdMethod::Phi`]. The Taylor evaluation path does not
    /// compute divided differences, so this setting may have no effect.
    ///
    /// # Arguments
    ///
    /// * `dd_method` - divided difference method
    pub fn with_dd_method(mut self, dd_method: LejaDdMethod) -> Self {
        self.dd_method = dd_method;
        self
    }

    /// Enable or disable re-use of Krylov information, passed to the evaluator.
    ///
    /// Default: `false`.
    ///
    /// # Arguments
    ///
    /// * `krylov_reuse` - `true` to re-use Krylov information
    pub fn with_krylov_reuse(mut self, krylov_reuse: bool) -> Self {
        self.krylov_reuse = krylov_reuse;
        self
    }

    /// Set the static spectrum bounds used to shift and scale the operator.
    ///
    /// Default: `a = -1`, `b = 0`, `c = 1`. Valid range: `a`, `b`, `c` finite,
    /// `a <= b` and `c >= 0`, checked by
    /// [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `a` - minimum real part of the spectrum
    /// * `b` - maximum real part of the spectrum
    /// * `c` - maximum magnitude of the imaginary part of the spectrum
    pub fn with_bounds(mut self, a: f64, b: f64, c: f64) -> Self {
        self.a = a;
        self.b = b;
        self.c = c;
        self
    }

    /// Validate the options and construct the Taylor evaluator.
    ///
    /// # Errors
    ///
    /// Returns an error if `tol` is not finite and positive, `m` is 0, or the
    /// spectrum bounds are invalid (not finite, `a > b`, or `c < 0`).
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

/// Choice of phi-function vector product evaluator for exponential integrators.
///
/// Default: `Krylov` with default [`KrylovOptions`].
#[derive(Clone, Debug)]
pub enum ExponentialEvaluator {
    /// Krylov subspace (Arnoldi) approximation.
    Krylov(KrylovOptions),
    /// Leja point polynomial interpolation.
    Leja(LejaOptions),
    /// Truncated Taylor series.
    Taylor(TaylorOptions),
}

impl Default for ExponentialEvaluator {
    fn default() -> Self {
        Self::Krylov(KrylovOptions::default())
    }
}

/// Builder for exponential time integrators.
///
/// Holds the initial condition, method, nonautonomous correction threshold and
/// phi-function evaluator choice. Construct the integrator with
/// [`ExponentialIntegratorBuilder::build`]. Default evaluator: Krylov with
/// default options; default `tol_fdt`: `1e-8`.
pub struct ExponentialIntegratorBuilder {
    t0: f64,
    y0: Mat<f64>,
    method: ExponentialMethod,
    tol_fdt: f64,
    evaluator: ExponentialEvaluator,
}

impl ExponentialIntegratorBuilder {
    /// Create a builder for an exponential integrator with default options.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state, an $n \times 1$ column; it is copied
    /// * `method` - the exponential integration method
    pub fn new(t0: f64, y0: MatRef<'_, f64>, method: ExponentialMethod) -> Self {
        Self {
            t0,
            y0: y0.to_owned(),
            method,
            tol_fdt: 1e-8,
            evaluator: ExponentialEvaluator::default(),
        }
    }

    /// Set the threshold for the nonautonomous time derivative correction.
    ///
    /// The time derivative of the right hand side is estimated by a forward
    /// finite difference. A negative value disables the correction. For
    /// `Epi2`, `Exprb2` and `Epi3` any value $\ge 0$ enables the correction;
    /// for `Exprb3` the correction is used only when the max-norm of the
    /// estimated derivative exceeds `tol_fdt`. Default: `1e-8`. Valid range:
    /// any finite value, checked by [`ExponentialIntegratorBuilder::build`].
    ///
    /// # Arguments
    ///
    /// * `tol_fdt` - threshold for the nonautonomous correction; negative to disable
    pub fn with_tol_fdt(mut self, tol_fdt: f64) -> Self {
        self.tol_fdt = tol_fdt;
        self
    }

    /// Set the phi-function evaluator.
    ///
    /// Default: `ExponentialEvaluator::Krylov` with default options.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - evaluator choice and its options
    pub fn with_evaluator(mut self, evaluator: ExponentialEvaluator) -> Self {
        self.evaluator = evaluator;
        self
    }

    /// Use the Krylov evaluator with the given options.
    ///
    /// # Arguments
    ///
    /// * `options` - Krylov evaluator options
    pub fn with_krylov(self, options: KrylovOptions) -> Self {
        self.with_evaluator(ExponentialEvaluator::Krylov(options))
    }

    /// Use the Leja evaluator with the given options.
    ///
    /// # Arguments
    ///
    /// * `options` - Leja evaluator options
    pub fn with_leja(self, options: LejaOptions) -> Self {
        self.with_evaluator(ExponentialEvaluator::Leja(options))
    }

    /// Use the Taylor evaluator with the given options.
    ///
    /// # Arguments
    ///
    /// * `options` - Taylor evaluator options
    pub fn with_taylor(self, options: TaylorOptions) -> Self {
        self.with_evaluator(ExponentialEvaluator::Taylor(options))
    }

    /// Validate the configuration and construct the integrator.
    ///
    /// # Returns
    ///
    /// A boxed [`crate::ode_exprb::ExprbIntegrator`] for `Exprb3`, or a boxed
    /// [`crate::ode_epirk::EpirkIntegrator`] for `Epi2`, `Exprb2` and `Epi3`,
    /// using the selected evaluator.
    ///
    /// # Errors
    ///
    /// Returns an [`IntegratorBuildError`] if
    ///
    /// * `tol_fdt` is not finite, or
    /// * for a Krylov evaluator: `tol` is not finite and positive, `max_dim`
    ///   is 0, `m` is 0, or `m` is greater than `max_dim`, or
    /// * for a Leja evaluator: `tol` is not finite and positive, `m` is 0,
    ///   `m + 2` overflows, the spectrum bounds `a`, `b`, `c` are not finite,
    ///   `a > b` or `c < 0`, or (adaptive spectrum) `spec_tol` is not finite
    ///   and non-negative or `safety_factor` is not finite and positive, or
    /// * for a Taylor evaluator: `tol` is not finite and positive, `m` is 0,
    ///   or the bounds `a`, `b`, `c` are not finite, `a > b` or `c < 0`.
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

    /// Construct the integrator around an already constructed evaluator.
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
