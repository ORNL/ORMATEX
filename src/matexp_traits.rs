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
//! The phi-function evaluator traits.
//!
//! This module defines the interfaces used by the exponential integrators to
//! evaluate products of phi-functions with vectors,
//! $\varphi_k(t A) v$, where $\varphi_k$ is defined by
//!
//! $$ \varphi_0(z) = e^z, \qquad
//!    \varphi_k(z) = \int_0^1 e^{(1-\theta) z} \frac{\theta^{k-1}}{(k-1)!} \thinspace d\theta $$
//!
//! [`DensePhikvEvaluator`] is implemented by methods acting on a dense matrix
//! `A` (Pade, Taylor, Cauchy contour integral, ...), while
//! [`LinOpPhikvEvaluator`] is implemented by matrix-free methods acting on a
//! linear operator (Krylov, Leja, ...). [`PhikvStatus`] describes the
//! convergence status of an iterative evaluation.
//!
//! # References
//!
//! * M. Hochbruck, A. Ostermann, "Exponential integrators", Acta Numerica 19
//!   (2010) 209-286, doi:10.1017/S0962492910000048.
//! * M. Caliari, F. Cassini, F. Zivcovich, "BAMPHI: Chebyshev and rational
//!   approximations of phi-functions applied to vectors", J. Comput. Appl.
//!   Math. 423 (2023) 114973.
use crate::ode_sys::DynRefExtendedLinOp;
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer_traits::ComplexField;

/// Status report of an iterative phi-function vector product evaluation.
///
/// The fields are private and currently unused.
pub struct PhikvStatus {
    /// Whether the iteration converged
    _conv: bool,
    /// Number of internal iterations required
    _iter: usize,
    /// Error estimate at termination
    _err: f64,
}

/// Trait for implementors of a $\varphi_k(A \thinspace dt) v$ method for dense `A`.
///
/// The scalar type `T` defaults to `f64`. The time step `dt` is always a real
/// `f64`.
pub trait DensePhikvEvaluator<T: ComplexField = f64> {
    /// Evaluates the phi-function vector product $\varphi_k(dt \thinspace A) v_0$.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - time step scale factor
    /// * `v0` - the vector (or matrix of column vectors) to which $\varphi_k$ is applied
    /// * `k` - the phi-function order
    ///
    /// # Returns
    ///
    /// The product $\varphi_k(dt \thinspace A) v_0$, same size as `v0`.
    fn apply_phi_k(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>, k: usize) -> Mat<T>;

    /// Evaluates a linear combination of phi-function vector products.
    ///
    /// Computes
    ///
    /// $$ \sum_j \varphi_{k_j}(dt \thinspace A) \thinspace v_j $$
    ///
    /// where $k_j$ is `ks[j]` and $v_j$ is `vb[j]`. The default implementation
    /// calls [`DensePhikvEvaluator::apply_phi_k`] once per term in serial and
    /// sums the results. Implementors may override it with a more efficient
    /// evaluation.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - time step scale factor
    /// * `vb` - the vectors $v_j$ to which each phi-function is applied
    /// * `ks` - the phi-function orders $k_j$, one per entry of `vb`
    ///
    /// # Returns
    ///
    /// The summed product, with `a.nrows()` rows and `vb[0].ncols()` columns.
    ///
    /// # Panics
    ///
    /// Panics if `vb` is empty or if `vb` and `ks` differ in length.
    fn apply_phi_k_v(&self, a: MatRef<T>, dt: f64, vb: &Vec<MatRef<T>>, ks: &Vec<usize>) -> Mat<T>
    {
        assert!(!vb.is_empty());
        assert!(vb.len() == ks.len());
        let mut out = Mat::zeros(a.nrows(), vb[0].ncols());
        // default loops over each phi_k function in serial
        for (v, k) in vb.iter().zip(ks) {
            // if v.norm_l2() >= 0.0 {
            out += self.apply_phi_k(a, dt, v.as_ref(), *k);
        }
        out
    }

    /// Prepare for a subsequent `apply_*` call.
    ///
    /// Implementors may use this hook to precompute and cache data that
    /// depends only on `a` and `dt` (e.g. matrix factorizations). The default
    /// implementation does nothing.
    ///
    /// # Arguments
    ///
    /// * `_a` - dense square matrix $A$
    /// * `_dt` - time step scale factor
    /// * `_v0` - the vector to which the phi-function will be applied
    /// * `_k` - the phi-function order
    fn apply_prepare(&mut self, _a: MatRef<T>, _dt: f64, _v0: MatRef<T>, _k: usize) {
        // default is null-op
    }
}

/// Trait for implementors of a $\varphi_k(A \thinspace dt) v$ method for sparse or
/// matrix-free linear operator `A`.
///
/// The scalar type `T` defaults to `f64`.
pub trait LinOpPhikvEvaluator<T: ComplexField = f64> {
    /// Evaluates a linear combination of phi-function vector products.
    ///
    /// Computes
    ///
    /// $$ \sum_{j} \varphi_j(dt \thinspace A) \thinspace v_j $$
    ///
    /// where $v_j$ is `vb[j]`.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - the extended linear operator wrapping $A$ and the vectors `vb`
    /// * `dt` - time step scale factor
    /// * `vb` - the vectors $v_j$ to which the phi-function of order $j$ is applied
    ///
    /// # Returns
    ///
    /// The summed product, a matrix with as many rows as `vb[0]`.
    fn apply_phi_k_v(
        &mut self,
        a_lo: &DynRefExtendedLinOp,
        dt: f64,
        vb: &Vec<MatRef<T>>,
    ) -> Mat<T>;

    /// Evaluates the phi-function vector product $\varphi_k(dt \thinspace A) v$.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - linear operator $A$
    /// * `dt` - time step scale factor
    /// * `v` - the vector to which $\varphi_k$ is applied
    /// * `k` - the phi-function order
    ///
    /// # Returns
    ///
    /// The product $\varphi_k(dt \thinspace A) v$, same size as `v`.
    fn apply_phi_k(&self, a_lo: &dyn LinOp<T>, dt: f64, v: MatRef<T>, k: usize) -> Mat<T>;

    /// Prepare for a subsequent `apply_*` call.
    ///
    /// When `ext` is `Some((ext_a_lo, vb))` the implementation should compute the
    /// $p = $ `vb.len() - 1` Taylor-block iterates
    /// $w_j = A_{ext}^j \tilde{v}$ ($j = 1, \ldots, p$), where `ext_a_lo` is
    /// $A_{ext}$, use the upper block of $w_p$ as the Arnoldi starting vector
    /// (correct per BAMPHI section 3), and cache the iterates for
    /// zero-duplication reuse in the subsequent `apply_phi_k_v` call.
    ///
    /// When `ext` is `None` the legacy path is used: `v` is the Arnoldi starting
    /// vector and `k` is the zero-prefix length. The default implementation does
    /// nothing.
    ///
    /// # Arguments
    ///
    /// * `_a_lo` - linear operator $A$
    /// * `_dt` - time step scale factor
    /// * `_v` - the vector to which the phi-function will be applied
    /// * `_k` - the phi-function order (zero-prefix length in the legacy path)
    /// * `_ext` - optional extended linear operator and the vectors `vb`
    fn apply_prepare(
        &mut self,
        _a_lo: &dyn LinOp<T>,
        _dt: f64,
        _v: MatRef<T>,
        _k: usize,
        _ext: Option<(&DynRefExtendedLinOp, &Vec<MatRef<T>>)>,
    ) {
        // default is null-op
    }
}
