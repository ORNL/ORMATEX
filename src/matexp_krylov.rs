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
// Krylov Matrix Exponential Methods
//
use faer::prelude::*;
use faer::matrix_free::LinOp;
use crate::arnoldi::arnoldi_lop;
use crate::ode_sys::ExtendedLinOp;
use crate::matexp_pade;
use crate::matexp_pade::DensePhikvEvaluator;


/// Krylov methods to compute Sparse Matrix Exponential
/// and Phi functions
pub struct KrylovExpm {
    /// dense matrix exponential and phi function evaluator
    expmv: Box<dyn DensePhikvEvaluator>,
    /// max krylov dim size
    krylov_dim: usize,
    /// incomplete ortho depth
    iom: usize,
}

impl KrylovExpm {
    pub fn new(expmv: Box<dyn DensePhikvEvaluator>, krylov_dim: usize, iom_in: Option<usize>) -> Self {
        assert!(krylov_dim > 0);
        Self {
            expmv,
            krylov_dim,
            iom: iom_in.unwrap_or(2),
        }
    }

    /// Computes exp(A*dt)*v0 when A is a linear operator
    pub fn apply_linop(&self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>)
        -> Mat<f64>
    {
        let (q, h, _b) = arnoldi_lop(a_lo, dt, v0.as_ref(), self.krylov_dim, self.iom);
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        // let matexp = matexp_pade::matexp(h.as_ref(), 1.0);
        // return faer::Scale(beta) * (q.as_ref() * matexp.as_ref() * unit_vec)
        return faer::Scale(beta) * (q.as_ref() * self.expmv.phik_apply(h.as_ref(), 1.0, unit_vec.as_ref(), 0))
    }

    /// Computes phi_k(A*dt) * v0 where A is a LinOp
    pub fn apply_phi_linop(
        &self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>, k: usize)
        -> Mat<f64>
    {
        let (q, h, _b) = arnoldi_lop(a_lo, 1.0, v0.as_ref(), self.krylov_dim, self.iom);
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        // let phi_k = matexp_pade::phi_ext((faer::Scale(dt) * h.as_ref()).as_ref(), k);
        // return faer::Scale(beta) * (q.as_ref() * phi_k.as_ref() * unit_vec)
        return faer::Scale(beta) * (q.as_ref() * self.expmv.phik_apply(h.as_ref(), dt, unit_vec.as_ref(), k))
    }

    /// Computes tripplet (phi_k(A*dt)*v0, phi_k(A*dt*2)*v0, phi_k(A*dt*3)*v0)
    /// where A is a LinOp
    /// This saves two calls to arnoldi.
    ///
    /// From ref:  M. Hochbruck, C. Lubich and H. Selhofer.
    /// Exponential Integrators for Large
    /// Systems of Differential Equations.  J. Sci. Comp. 1996.
    pub fn apply_phi_linop_3(
        &self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>, k: usize)
        -> (Mat<f64>, Mat<f64>, Mat<f64>)
    {
        let (q, h, _b) = arnoldi_lop(a_lo, 1.0, v0.as_ref(), self.krylov_dim, self.iom);
        let phi_k = matexp_pade::phi_ext((faer::Scale(dt) * h.as_ref()).as_ref(), k);
        let id = faer::Mat::<f64>::identity(phi_k.nrows(), phi_k.ncols());
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(phi_k.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        // compute unscaled phi_k_1 = phi_k(A*dt)*v0
        let phi_k_1 = faer::Scale(beta) * (q.as_ref() * phi_k.as_ref() * unit_vec.as_ref());
        // compute scaled phi_k_2 = phi_k(A*dt*2)*v0
        let phi_2tau_h = (faer::Scale(dt * 1./2.) * h.as_ref() * phi_k.as_ref() + id.as_ref()) * phi_k.as_ref();
        let phi_k_2 = q.as_ref()
            * phi_2tau_h.as_ref()
            * unit_vec.as_ref() * faer::Scale(beta);
        // compute scaled phi_k_3 = phi_k(A*dt*3)*v0
        let phi_k_3 = q.as_ref()
            * (faer::Scale(2./3.) * (faer::Scale(dt) * h.as_ref() * phi_k.as_ref() + id.as_ref())
            * phi_2tau_h.as_ref() + faer::Scale(1./3.) * phi_k.as_ref())
            * unit_vec.as_ref() * faer::Scale(beta);
        (phi_k_1, phi_k_2, phi_k_3)
    }

    /// Similar to apply_phi_linop_3, this method evaluates linear combinations
    /// of phi functions using only a single matexp call, thus reducing the
    /// number of calls to arnoldi.
    ///
    /// S. Gaudreault, G. Rainwater, and M. Tokman.
    /// "KIOPS: A fast adaptive Krylov subspace solver for exponential integrators."
    /// Journal of Computational Physics 372 (2018): 236-255.
    ///
    /// TODO: Implement krylov adaptivity.
    ///
    /// Args:
    /// * `a_lo` - Linear operator, A, in [phi_0(A*dt) * v_0 + phi_1(A*dt) * v_1 + ...]
    /// * `dt` - time step scale.
    /// * `vb` - Vec of rhs, [v] in [phi_0(A*dt) * v_0 + ...]
    pub fn kiops_fixedsteps<'a>(
        &self,
        // a_lo: Box<dyn LinOp<f64> + 'a>,
        ext_a_lo: &ExtendedLinOp,
        dt: f64,
        vb: &Vec<MatRef<f64>>)
        -> Mat<f64>
    {
        // setup the extended rhs vector
        let (ext_v, n) = ext_a_lo.get_v(vb);

        // compute phi_0(dt*A_ext)*v_ext
        let w = self.apply_linop(&ext_a_lo, dt, ext_v.as_ref());

        // extract first n rows
        w.get(0..n, 0..1).to_owned()
    }
}


#[cfg(test)]
mod test_matexp_krylov {
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

}
