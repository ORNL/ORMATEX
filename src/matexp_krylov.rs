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
use std::cmp::{max, min};
use faer::prelude::*;
use faer::matrix_free::LinOp;
use crate::arnoldi::{arnoldi_lop, arnoldi_lop_restarted};
use crate::ode_sys::{DynRefExtendedLinOp};
use crate::matexp_pade;
use crate::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};


/// Krylov methods to compute Sparse Matrix Exponential
/// and Phi functions
pub struct KrylovExpm {
    /// dense matrix exponential and phi function evaluator
    expmv: Box<dyn DensePhikvEvaluator>,
    /// current krylov dim
    m: usize,
    /// max krylov dim size
    krylov_dim: usize,
    /// incomplete ortho depth
    iom: usize,
    /// storage for tmp hessenberg
    hs: Mat<f64>,
    qs: Mat<f64>,
    /// tolerance
    tol: f64,
    /// verbosity
    verbose: bool,
}

impl KrylovExpm {
    pub fn new(expmv: Box<dyn DensePhikvEvaluator>, m: usize, max_krylov_dim: usize, tol: f64, iom_in: Option<usize>) -> Self {
        assert!(max_krylov_dim > 0);
        assert!(m <= max_krylov_dim);
        Self {
            expmv,
            m: min(m, max_krylov_dim),
            krylov_dim: max_krylov_dim,
            hs: faer::Mat::zeros(max_krylov_dim, max_krylov_dim),
            qs: faer::Mat::zeros(max_krylov_dim, max_krylov_dim),
            iom: iom_in.unwrap_or(2),
            tol: tol,
            verbose: false,
        }
    }

    /// Set extra verbosity for additional stdout output
    pub fn set_verbosity(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Computes exp(A*dt)*v0 when A is a linear operator.
    /// Alias to apply_phik_linop with k=0.
    ///
    /// Args:
    /// * `a_lo` - Linear operator, A
    /// * `dt` - time step scale.
    /// * `v0` - the vector to which the matrix exponential is applied
    ///
    pub fn apply_linop(&mut self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>)
        -> Mat<f64>
    {
        self.apply_phik_linop(a_lo, dt, v0, 0)
    }

    /// Computes phi_k(A*dt) * v0 where A is a LinOp and
    /// adapts the krylov dimension.
    ///
    /// Args:
    /// * `a_lo` - Linear operator, A
    /// * `dt` - time step scale.
    /// * `v0` - the vector to which the matrix phi-function is applied
    /// * `k` - the phi function order
    pub fn apply_phik_linop_adapt(
        &mut self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>, k: usize)
        -> Mat<f64>
    {
        log::info!("=== Adaptive KrylovExpm");
        // Allocate storage matrices with correct dimensions
        // The storage must be large enough to hold:
        // - hs: square matrix at least (m+1) x (m+1) where m can grow up to krylov_dim
        // - qs: (v0.nrows()) x (m+1) where m can grow up to krylov_dim
        let v0_dim = v0.nrows();
        let storage_size = self.krylov_dim + 1; // +1 for extended Hessenberg
        self.hs = faer::Mat::zeros(storage_size, storage_size);
        self.qs = faer::Mat::zeros(v0_dim, storage_size);

        // clear tmp hessenberg storage
        self.hs.fill(0.0);
        self.qs.fill(0.0);

        // run initial arnoldi iterations up to the current krylov dim
        let (mut breakdown_flag, mut breakdown_m) = arnoldi_lop_restarted(
            a_lo, dt, v0, self.hs.as_mut(), self.qs.as_mut(),
            0, self.m, self.iom);

        const BUFFER_M: usize = 2;
        const INCREMENT_M: usize = 20;
        let beta = v0.norm_l2();
        let mut res = v0.to_owned();
        let mut converged = false;
        let mut adapt_iter = 1;
        while !converged {
            // trim hessenberg to size
            let h_dim = min(self.m, breakdown_m);
            // get H_{m+1} view
            let h = self.hs.get(0..h_dim+1, 0..h_dim+1);
            let q = self.qs.get(.., 0..h_dim+1);

            // compute the dense matrix exponential of the hessenberg
            let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
            unit_vec[(0, 0)] = 1.0;
            let phi_h = self.expmv.phik_apply(
                h.as_ref(), 1.0, unit_vec.as_ref(), k);
            res = faer::Scale(beta) * (q.as_ref() * phi_h.as_ref());

            // compute error estimate
            let last_m = phi_h.nrows()-1;
            for p in (1..=min(10, last_m)).rev() {
                let last_m_p = last_m+1 - p;
                let final_updates = q.get(.., last_m_p..last_m+1) * phi_h.col(0).get(last_m_p..last_m+1);
                let err_est_p = final_updates.norm_l2();
                converged = self.tol > err_est_p;

                // log error estimate to stdout and log file
                if self.verbose {
                    let final_update = phi_h[(last_m, 0)] * (q.col(last_m));
                    let err_est_m = final_update.norm_l2();
                    println!("i: {adapt_iter}, m: {last_m}, mp: {last_m_p}, e_mp: {:.5e}, e_m: {:.5e} conv: {converged}, bdwn: {breakdown_flag}", err_est_p, err_est_m);
                }
                log::info!("adapt i: {adapt_iter}, mp: {last_m_p}, err: {:.6e}, converged: {converged}, bkdwn: {breakdown_flag}", err_est_p);

                // update krylov dim
                if converged {
                    let m_next = last_m_p + BUFFER_M;
                    self.m = m_next;
                    break;
                }
            }

            if !converged {
                // run arnoldi an additional INCREMENT_M iters
                let (bd, bd_n) = arnoldi_lop_restarted(
                    a_lo, dt, v0, self.hs.as_mut(), self.qs.as_mut(),
                    self.m, INCREMENT_M, self.iom);
                breakdown_m = bd_n;
                breakdown_flag = bd;
                // extend krylov dim
                self.m += INCREMENT_M;
            }

            // TODO: return Err() or Warning
            if self.m >= self.krylov_dim {
                self.m = self.krylov_dim;
                break
            }
            adapt_iter += 1;
        }

        // return final approximation beta*Q*exp(H)*e1
        res
    }

    /// Computes phi_k(A*dt) * v0 where A is a LinOp
    ///
    /// Args:
    /// * `a_lo` - Linear operator, A
    /// * `dt` - time step scale.
    /// * `v0` - the vector to which the matrix phi-function is applied
    /// * `k` - the phi function order
    pub fn apply_phik_linop(
        &self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>, k: usize)
        -> Mat<f64>
    {
        let (q, h, _b) = arnoldi_lop(a_lo, 1.0, v0.as_ref(), self.krylov_dim, self.iom);
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        return faer::Scale(beta) * (q.as_ref() * self.expmv.phik_apply(h.as_ref(), dt, unit_vec.as_ref(), k))
    }

    /// This method evaluates linear combinations
    /// of phi functions using only a single matexp call, thus reducing the
    /// number of calls to arnoldi.
    ///
    /// S. Gaudreault, G. Rainwater, and M. Tokman.
    /// "KIOPS: A fast adaptive Krylov subspace solver for exponential integrators."
    /// Journal of Computational Physics 372 (2018): 236-255.
    ///
    /// NOTE: Currently krylov apply_phik_linop_adapt implements an
    /// adptive krylov subspace dimension procedure via the
    /// error estimate noted in the reference.
    /// TODO: Implement substepping adaptivity.
    ///
    /// Args:
    /// * `ext_a_lo` - Linear operator, A, in [phi_0(A*tau) * v_0 + phi_1(A*tau) * v_1 + ...]
    /// * `tau` - time step scale.
    /// * `vb` - Vec of rhs, [v0, ..vn] in
    ///          [phi_0(A*tau)*v_0 + ..., phi_n(A*tau)*v_n]
    ///
    pub fn apply_linop_ext(
        &mut self,
        ext_a_lo: &DynRefExtendedLinOp,
        tau: f64,
        vb: &Vec<MatRef<f64>>)
        -> Mat<f64>
    {
        // setup the extended rhs vector
        let (ext_v, n) = ext_a_lo.get_v(vb);

        // compute phi_0(tau*A_ext)*v_ext with adaptive krylov dimension
        let w = self.apply_phik_linop_adapt(&ext_a_lo, tau, ext_v.as_ref(), 0);

        // extract first n rows
        w.get(0..n, 0..1).to_owned()
    }
}

impl LinOpPhikvEvaluator for KrylovExpm {
    fn apply_phi_k_v(&mut self, ext_a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64> {
        self.apply_linop_ext(ext_a_lo, dt, vb)
    }

    fn apply_phi_k(&self, a_lo: &dyn LinOp<f64>, dt: f64, v: MatRef<f64>, k: usize) -> Mat<f64> {
        self.apply_phik_linop(a_lo, dt, v, k)
    }
}


#[cfg(test)]
mod test_matexp_krylov {
    use assert_approx_eq::assert_approx_eq;
    use crate::mat_utils::mat_mat_approx_eq;
    use crate::matexp_pade::{matexp, phi_ext};
    use crate::test_common::{gen_test_b, gen_test_c};

    // bring everything from above (parent) module into scope
    use super::*;

    fn _run_krylov_phikv(test_b: Mat<f64>, test_v: Mat<f64>) {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // krylov method.
        let iom = 2;
        let m = 10;
        let tol = 1e-12;
        let max_krylov_dim = 100;
        let expmv = Box::new(matexp_pade::PadeExpm::new(12));
        let mut krylov_phikv_eval = KrylovExpm::new(expmv, m, max_krylov_dim, tol, Some(iom));
        krylov_phikv_eval.set_verbosity(true);

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref(),];

        // compute phi_0(dt*A)*b0
        let dt = 0.3;
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi0mv_krylov_pm: Mat<f64> = krylov_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi0mv_pade_dense = matexp(test_b.as_ref(), dt) * test_v.as_ref();
        println!("krylov phi0mv: {:?}", &phi0mv_krylov_pm);
        println!("pade phi0mv: {:?}", &phi0mv_pade_dense);
        mat_mat_approx_eq(
            phi0mv_krylov_pm.as_ref(), phi0mv_pade_dense.as_ref(), 1e-8);


        // compute phi_1(dt*A)*b0
        let zeros = Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), test_v.as_ref(),];
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi1mv_krylov_pm: Mat<f64> = krylov_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi1mv_pade_dense = phi_ext((dt*test_b).as_ref(), 1) * test_v.as_ref();
        println!("krylov phi1mv: {:?}", &phi1mv_krylov_pm);
        println!("pade phi1mv: {:?}", &phi1mv_pade_dense);
        mat_mat_approx_eq(
            phi1mv_krylov_pm.as_ref(), phi1mv_pade_dense.as_ref(), 1e-8);
    }

    #[test]
    fn test_krylov_phikv_small() {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // krylov method for a small 3x3 A
        let (test_b, test_v) = gen_test_b();
        _run_krylov_phikv(test_b, test_v);
    }

    #[test]
    fn test_krylov_phikv_large() {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // krylov method for a larger 80x80 A
        let (test_b, test_v) = gen_test_c(80);
        let scale = 20.0;  // increase stiffness of the problem
        _run_krylov_phikv(faer::Scale(scale) * test_b, test_v);
    }
}
