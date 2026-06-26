use std::error::Error;

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
/// The phi-function evaluator traits
use faer::prelude::*;
use faer::matrix_free::LinOp;
use crate::ode_sys::{DynRefExtendedLinOp};


pub struct PhikvStatus {
    /// converged status
    conv: bool,
    /// number of internal iterations required
    iter: usize,
    /// err estimate
    err: f64,
}


/// Trait for implementors of a phi_k(A*dt)*v method for dense A
/// Provides capability to evaluate a phi_k function-vector product.
pub trait DensePhikvEvaluator
{
    /// Evaluates phi_k(dt*A) * v0
    fn phik_apply(&self, a: MatRef<f64>, dt: f64, v0: MatRef<f64>, k: usize) -> Mat<f64>;
}


/// Trait for implementors of a phi_k(A*dt)*v method for Sparse or LinOp A
pub trait LinOpPhikvEvaluator
{
    /// Evaluate a linear combination of phi-function vector prodcuts
    /// of the form [phi_0(dt*A) * v0 + phi_1(dt*A) * v1 + ... phi_k(dt*A) * vk]
    fn apply_phi_k_v(&mut self, a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64>;

    /// Evaluate the phi-function vector prodcut:
    /// phi_k(dt*A) * vk
    fn apply_phi_k(&self, a_lo: &dyn LinOp<f64>, dt: f64, v: MatRef<f64>, k: usize) -> Mat<f64>;

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
        a_lo: &dyn LinOp<f64>,
        dt: f64,
        v: MatRef<f64>,
        k: usize,
        ext: Option<(&DynRefExtendedLinOp, &Vec<MatRef<f64>>)>,
    ) {
        // default is null-op
    }
}
