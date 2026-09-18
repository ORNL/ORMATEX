/*
 * Copyright© 2025 UT-Battelle, LLC
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
use crate::ode_sys::DynRefExtendedLinOp;
use faer::matrix_free::LinOp;
/// The phi-function evaluator traits
use faer::prelude::*;
use faer_traits::ComplexField;

pub struct PhikvStatus {
    /// converged status
    _conv: bool,
    /// number of internal iterations required
    _iter: usize,
    /// err estimate
    _err: f64,
}

/// Trait for implementors of a phi_k(A*dt)*v method for dense A.
pub trait DensePhikvEvaluator<T: ComplexField = f64> {
    /// Evaluates phi_k(dt*A) * v0
    fn apply_phi_k(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>, k: usize) -> Mat<T>;

    /// Evaluate a linear combination of phi-function vector prodcuts
    /// of the form [phi_0(dt*A) * v0 + phi_1(dt*A) * v1 + ... phi_k(dt*A) * vk]
    fn apply_phi_k_v(&self, a: MatRef<T>, dt: f64, vb: &Vec<MatRef<T>>, ks: &Vec<usize>) -> Mat<T>
    {
        let mut out = Mat::zeros(a.nrows(), 1);
        // default loops over each phi_k function in serial
        for (v, k) in vb.iter().zip(ks) {
            // if v.norm_l2() >= 0.0 {
            out += self.apply_phi_k(a, dt, v.as_ref(), *k);
        }
        out
    }

    /// Prepare for apply_*.
    fn apply_prepare(&mut self, _a: MatRef<T>, _dt: f64, _v0: MatRef<T>, _k: usize) {
        // default is null-op
    }
}

/// Trait for implementors of a phi_k(A*dt)*v method for Sparse or LinOp A
pub trait LinOpPhikvEvaluator<T: ComplexField = f64> {
    /// Evaluate a linear combination of phi-function vector prodcuts
    /// of the form [phi_0(dt*A) * v0 + phi_1(dt*A) * v1 + ... phi_k(dt*A) * vk]
    fn apply_phi_k_v(
        &mut self,
        a_lo: &DynRefExtendedLinOp,
        dt: f64,
        vb: &Vec<MatRef<T>>,
    ) -> Mat<T>;

    /// Evaluate the phi-function vector prodcut:
    /// phi_k(dt*A) * vk
    fn apply_phi_k(&self, a_lo: &dyn LinOp<T>, dt: f64, v: MatRef<T>, k: usize) -> Mat<T>;

    /// Prepare for apply_*.
    ///
    /// When `ext` is `Some((ext_a_lo, vb))` the implementation should compute the
    /// p = vb.len()-1 Taylor-block iterates `w_j = ext_a_lo^j · tilde_v` (j=1..=p),
    /// use `upper_block(w_p)` as the Arnoldi starting vector (correct per BAMPHI §3),
    /// and cache the iterates for zero-duplication reuse in the subsequent
    /// `apply_phi_k_v` call.
    ///
    /// When `ext` is `None` the legacy path is used: `v` is the Arnoldi starting
    /// vector and `k` is the zero-prefix length.
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
