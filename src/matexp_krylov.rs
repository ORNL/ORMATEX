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
//! Krylov subspace matrix exponential and phi-function methods for linear operators.
//!
//! [`KrylovExpm`] approximates phi-function vector products $\varphi_k(A \thinspace dt) v$
//! for a matrix-free linear operator $A$ by projecting onto the Krylov subspace
//! $\mathcal K_m(A, v) = \mathrm{span}\lbrace v, Av, \ldots, A^{m-1}v\rbrace $ with an
//! Arnoldi iteration. With the Arnoldi relation $A Q_m = Q_{m+1} \bar H_m$,
//! the approximation is
//!
//! $$ \varphi_k(A \thinspace dt) v \approx \Vert v\Vert_2 \thinspace Q \thinspace \varphi_k(dt \thinspace H) \thinspace e_1 $$
//!
//! where the small dense Hessenberg matrix function is evaluated by a
//! [`crate::matexp_traits::DensePhikvEvaluator`] (Pade, Taylor, Cauchy, ...).
//! The module provides a fixed dimension evaluation, an adaptive Krylov
//! dimension evaluation driven by an error estimate, and the evaluation of
//! linear combinations of phi-functions through an extended (augmented) linear
//! operator, which requires only a single exponential.
//! [`KrylovExpm`] implements [`crate::matexp_traits::LinOpPhikvEvaluator`].
//!
//! # References
//!
//! * Y. Saad, "Analysis of some Krylov subspace approximations to the matrix
//!   exponential operator", SIAM J. Numer. Anal. 29(1) (1992) 209-228.
//! * S. Gaudreault, G. Rainwater, M. Tokman, "KIOPS: A fast adaptive Krylov
//!   subspace solver for exponential integrators", J. Comput. Phys. 372 (2018)
//!   236-255, doi:10.1016/j.jcp.2018.06.026.
//! * M. Caliari, F. Cassini, F. Zivcovich, "BAMPHI: Chebyshev and rational
//!   approximations of phi-functions applied to vectors", J. Comput. Appl.
//!   Math. 423 (2023) 114973.
use crate::arnoldi::{arnoldi_lop, arnoldi_lop_restarted};
use crate::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use crate::ode_sys::DynRefExtendedLinOp;
use faer::matrix_free::LinOp;
use faer::prelude::*;
use std::cmp::{max, min};

/// Krylov methods to compute the sparse (matrix-free) matrix exponential
/// and phi-function vector products.
///
/// Holds the Krylov dimension state (current, maximum, adaptive increment and
/// lookback), the incomplete orthogonalization depth, the tolerance, and
/// work storage for the Arnoldi basis and Hessenberg matrix. The adaptive
/// evaluation updates the current Krylov dimension, so it persists between
/// calls.
pub struct KrylovExpm {
    /// dense matrix exponential and phi function evaluator
    expmv: Box<dyn DensePhikvEvaluator>,
    /// current krylov dim
    m: usize,
    /// max krylov dim size
    krylov_dim_max: usize,
    /// krylov dim increment used in adaptive krylov subspace method
    krylov_dim_inc: usize,
    /// number of krylov steps to lookback over in adaptive krylov logic
    krylov_dim_lookback: usize,
    /// incomplete ortho depth
    iom: usize,
    /// storage for tmp hessenberg
    hs: Mat<f64>,
    /// storage for tmp orthonormal krylov basis
    qs: Mat<f64>,
    /// tolerance
    tol: f64,
    /// verbosity
    verbose: bool,
}

impl KrylovExpm {
    /// Creates a new Krylov phi-function evaluator.
    ///
    /// The adaptive dimension increment defaults to 20 and the lookback
    /// to 10, and verbosity is off. See [`KrylovExpm::set_krylov_dim_inc`],
    /// [`KrylovExpm::set_krylov_dim_lookback`] and [`KrylovExpm::set_verbosity`].
    ///
    /// # Arguments
    ///
    /// * `expmv` - dense evaluator used for the phi-function of the small Hessenberg matrix
    /// * `m` - initial Krylov subspace dimension
    /// * `krylov_dim_max` - maximum Krylov subspace dimension
    /// * `tol` - tolerance on the error estimate of the adaptive method
    /// * `iom_in` - incomplete orthogonalization depth; defaults to 2 if `None`
    ///
    /// # Panics
    ///
    /// Panics if `krylov_dim_max` is zero or if `m > krylov_dim_max`.
    pub fn new(
        expmv: Box<dyn DensePhikvEvaluator>,
        m: usize,
        krylov_dim_max: usize,
        tol: f64,
        iom_in: Option<usize>,
    ) -> Self {
        assert!(krylov_dim_max > 0);
        assert!(m <= krylov_dim_max);
        Self {
            expmv,
            m: min(m, krylov_dim_max),
            krylov_dim_max: krylov_dim_max,
            krylov_dim_inc: 20,
            krylov_dim_lookback: 10,
            hs: faer::Mat::zeros(krylov_dim_max, krylov_dim_max),
            qs: faer::Mat::zeros(krylov_dim_max, krylov_dim_max),
            iom: iom_in.unwrap_or(2),
            tol: tol,
            verbose: false,
        }
    }

    /// Sets extra verbosity for additional stdout output.
    ///
    /// # Arguments
    ///
    /// * `verbose` - if true, the adaptive method prints error estimates
    pub fn set_verbosity(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Sets the adaptive Krylov dimension increment.
    ///
    /// # Arguments
    ///
    /// * `krylov_dim_inc` - number of Arnoldi iterations added per adaptive step
    pub fn set_krylov_dim_inc(&mut self, krylov_dim_inc: usize) {
        self.krylov_dim_inc = krylov_dim_inc;
    }

    /// Sets the adaptive Krylov dimension lookback.
    ///
    /// # Arguments
    ///
    /// * `krylov_dim_lookback` - number of trailing Krylov steps over which the
    ///   error estimate is examined (values below 2 are treated as 2)
    pub fn set_krylov_dim_lookback(&mut self, krylov_dim_lookback: usize) {
        self.krylov_dim_lookback = krylov_dim_lookback;
    }

    /// Computes $\exp(A \thinspace dt) v_0$ when `A` is a linear operator.
    ///
    /// Alias to [`KrylovExpm::apply_phik_linop`] with $k = 0$.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - Linear operator, $A$
    /// * `dt` - time step scale
    /// * `v0` - the vector to which the matrix exponential is applied
    ///
    /// # Returns
    ///
    /// The approximation of $\exp(A \thinspace dt) v_0$.
    pub fn apply_linop(&mut self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>) -> Mat<f64> {
        self.apply_phik_linop(a_lo, dt, v0, 0)
    }

    /// Computes $\varphi_k(A \thinspace dt) v_0$ where `A` is a linear operator, adapting
    /// the Krylov dimension.
    ///
    /// Runs Arnoldi iterations up to the current Krylov dimension, evaluates
    /// the approximation $\Vert v_0\Vert_2 \thinspace Q \thinspace \varphi_k(H) \thinspace e_1$ using the leading
    /// $(m+1) \times (m+1)$ block of the Hessenberg matrix, and estimates the
    /// error from the norm of the contribution of the last $p$ Krylov basis
    /// vectors, $p = 1, \ldots, $ `krylov_dim_lookback`. If the estimate does not
    /// meet the tolerance, more Arnoldi iterations are added
    /// (by at most `krylov_dim_inc`, limited by the available storage) and the
    /// evaluation is repeated. When converged, the stored Krylov dimension
    /// is reset to the smallest tail dimension that met the tolerance plus a
    /// buffer of 2, for use by later calls. The loop also terminates once the
    /// maximum Krylov dimension is reached, in which case the result may not
    /// satisfy the tolerance and no error is returned.
    ///
    /// This method mutates the stored Krylov dimension and work storage,
    /// and prints a summary (convergence, dimension, error estimate and
    /// timing) to stdout.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - Linear operator, $A$
    /// * `dt` - time step scale
    /// * `v0` - the vector to which the matrix phi-function is applied
    /// * `k` - the phi function order
    ///
    /// # Returns
    ///
    /// The approximation of $\varphi_k(A \thinspace dt) v_0$.
    pub fn apply_phik_linop_adapt(
        &mut self,
        a_lo: &dyn LinOp<f64>,
        dt: f64,
        v0: MatRef<f64>,
        k: usize,
    ) -> Mat<f64> {
        let clock = std::time::Instant::now();
        log::info!("=== Adaptive KrylovExpm");
        println!("=== Adaptive KrylovExpm");
        // Allocate storage matrices with correct dimensions
        // The storage must be large enough to hold:
        // - hs: square matrix at least (m+1) x (m+1) where m can grow up to krylov_dim_max
        // - qs: (v0.nrows()) x (m+1) where m can grow up to krylov_dim_max
        let v0_dim = v0.nrows();
        let storage_size = self.krylov_dim_max + 1; // +1 for extended Hessenberg
        self.hs = faer::Mat::zeros(storage_size, storage_size);
        self.qs = faer::Mat::zeros(v0_dim, storage_size);

        // clear tmp hessenberg storage
        self.hs.fill(0.0);
        self.qs.fill(0.0);

        // run initial arnoldi iterations up to the current krylov dim
        let (mut breakdown_flag, mut breakdown_m) = arnoldi_lop_restarted(
            a_lo,
            dt,
            v0,
            self.hs.as_mut(),
            self.qs.as_mut(),
            0,
            self.m,
            self.iom,
        );

        const BUFFER_M: usize = 2;
        let beta = v0.norm_l2();
        let mut res = v0.to_owned();
        let mut converged = false;
        let mut adapt_iter = 1;
        let mut err_est_p = 0.0;
        while !converged {
            // trim hessenberg to size
            let h_dim = min(self.m, breakdown_m);
            // get H_{m+1} view
            let h = self.hs.get(0..h_dim + 1, 0..h_dim + 1);
            let q = self.qs.get(.., 0..h_dim + 1);

            // compute the dense matrix exponential of the hessenberg
            let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
            unit_vec[(0, 0)] = 1.0;
            let phi_h = self
                .expmv
                .apply_phi_k(h.as_ref(), 1.0, unit_vec.as_ref(), k);
            res = faer::Scale(beta) * (q.as_ref() * phi_h.as_ref());

            // compute error estimate
            let last_m = phi_h.nrows() - 1;
            let mut last_m_p = last_m;
            let krylov_dim_lookback = max(self.krylov_dim_lookback, 2);
            // p is the lookback
            for p in 1..=min(krylov_dim_lookback, last_m) {
                converged = p > 1;
                last_m_p = last_m + 1 - p;
                let final_updates =
                    q.get(.., last_m_p..last_m + 1) * phi_h.col(0).get(last_m_p..last_m + 1);
                err_est_p = final_updates.norm_l2();

                // log error estimate to stdout and log file
                if self.verbose {
                    let final_update = phi_h[(last_m, 0)] * (q.col(last_m));
                    let err_est_m = final_update.norm_l2();
                    println!("i: {adapt_iter}, m: {last_m}, mp: {last_m_p}, e_mp: {:.5e}, e_m: {:.5e} conv: {converged}, bdwn: {breakdown_flag}", err_est_p, err_est_m);
                }
                log::info!("adapt i: {adapt_iter}, mp: {last_m_p}, err: {:.6e}, converged: {converged}, bkdwn: {breakdown_flag}", err_est_p);

                if !(self.tol > err_est_p) {
                    break;
                }
            }
            if converged {
                let m_next = last_m_p + BUFFER_M;
                self.m = m_next;
            } else {
                // run arnoldi additional iters
                // cap the increment to available storage
                let storage_size = self.krylov_dim_max + 1;
                let max_increment = if storage_size > self.m {
                    storage_size - self.m - 1
                } else {
                    0
                };
                let dim_inc = min(self.krylov_dim_inc, max_increment);
                if dim_inc > 0 {
                    let (bd, bd_n) = arnoldi_lop_restarted(
                        a_lo,
                        dt,
                        v0,
                        self.hs.as_mut(),
                        self.qs.as_mut(),
                        self.m,
                        dim_inc,
                        self.iom,
                    );
                    breakdown_m = bd_n;
                    breakdown_flag = bd;
                    // extend krylov dim
                    self.m += dim_inc;
                }
            }

            // TODO: return Err() or Warning
            if self.m >= self.krylov_dim_max {
                self.m = self.krylov_dim_max;
                break;
            }
            adapt_iter += 1;
        }

        println!(
            "converged: {converged}, m: {}, err_est: {:0.6e}",
            self.m, err_est_p
        );
        println!("Krylov expmv time (s): {}", clock.elapsed().as_secs_f64());
        // return final approximation beta*Q*exp(H)*e1
        res
    }

    /// Computes $\varphi_k(A \thinspace dt) v_0$ where `A` is a linear operator.
    ///
    /// Uses a fixed Krylov dimension (the current dimension `m`) and no
    /// error control. The Arnoldi iteration is run on $A$ and the time step
    /// `dt` is applied in the dense evaluation of the Hessenberg matrix
    /// phi-function.
    ///
    /// # Arguments
    ///
    /// * `a_lo` - Linear operator, $A$
    /// * `dt` - time step scale
    /// * `v0` - the vector to which the matrix phi-function is applied
    /// * `k` - the phi function order
    ///
    /// # Returns
    ///
    /// The approximation of $\varphi_k(A \thinspace dt) v_0$.
    pub fn apply_phik_linop(
        &self,
        a_lo: &dyn LinOp<f64>,
        dt: f64,
        v0: MatRef<f64>,
        k: usize,
    ) -> Mat<f64> {
        let (q, h, _b) = arnoldi_lop(a_lo, 1.0, v0.as_ref(), self.m, self.iom);
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        return faer::Scale(beta)
            * (q.as_ref() * self.expmv.apply_phi_k(h.as_ref(), dt, unit_vec.as_ref(), k));
    }

    /// Evaluates a linear combination of phi-functions applied to vectors.
    ///
    /// Computes
    ///
    /// $$ \sum_{j=0}^{n} \varphi_j(\tau A) \thinspace v_j $$
    ///
    /// using only a single phi-function (matrix exponential) evaluation on an
    /// extended (augmented) linear operator, thus reducing the number of
    /// calls to Arnoldi. The extended right hand side is built by
    /// `ext_a_lo.get_v(vb)`, $\varphi_0(\tau A_{ext})$ is applied to it with
    /// [`KrylovExpm::apply_phik_linop_adapt`], and the first $n_{rows}$ rows of
    /// the result (the size of `vb[0]`) are returned.
    ///
    /// NOTE: Currently `apply_phik_linop_adapt` implements an
    /// adaptive Krylov subspace dimension procedure via the
    /// error estimate noted in the reference.
    /// TODO: Implement substepping adaptivity.
    ///
    /// # Arguments
    ///
    /// * `ext_a_lo` - Extended linear operator built from $A$ and the vectors
    ///   $v_1, \ldots, v_n$. It carries its own time scale, so `tau` is normally 1.0.
    /// * `tau` - additional time step scale applied to `ext_a_lo`
    /// * `vb` - Vec of right hand sides $[v_0, \ldots, v_n]$ in
    ///   $\sum_{j=0}^{n} \varphi_j(A \tau) v_j$
    ///
    /// # Returns
    ///
    /// A column vector with the same number of rows as `vb[0]`.
    ///
    /// # References
    ///
    /// * S. Gaudreault, G. Rainwater, M. Tokman, "KIOPS: A fast adaptive Krylov
    ///   subspace solver for exponential integrators", J. Comput. Phys. 372
    ///   (2018) 236-255, doi:10.1016/j.jcp.2018.06.026.
    /// * M. Caliari, F. Cassini, F. Zivcovich, "BAMPHI: Chebyshev and rational
    ///   approximations of phi-functions applied to vectors", J. Comput. Appl.
    ///   Math. 423 (2023) 114973.
    pub fn apply_linop_ext(
        &mut self,
        ext_a_lo: &DynRefExtendedLinOp,
        tau: f64,
        vb: &Vec<MatRef<f64>>,
    ) -> Mat<f64> {
        // setup the extended rhs vector
        let (ext_v, n) = ext_a_lo.get_v(vb);

        // compute phi_0(tau*A_ext)*v_ext with adaptive krylov dimension
        let w = self.apply_phik_linop_adapt(&ext_a_lo, tau, ext_v.as_ref(), 0);

        // extract first n rows
        w.get(0..n, 0..1).to_owned()
    }
}

impl LinOpPhikvEvaluator for KrylovExpm {
    fn apply_phi_k_v(
        &mut self,
        ext_a_lo: &DynRefExtendedLinOp,
        dt: f64,
        vb: &Vec<MatRef<f64>>,
    ) -> Mat<f64> {
        self.apply_linop_ext(ext_a_lo, dt, vb)
    }

    fn apply_phi_k(&self, a_lo: &dyn LinOp<f64>, dt: f64, v: MatRef<f64>, k: usize) -> Mat<f64> {
        self.apply_phik_linop(a_lo, dt, v, k)
    }
}

#[cfg(test)]
mod test_matexp_krylov {
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
        let krylov_dim_max = 100;
        let expmv = Box::new(crate::matexp_pade::PadeExpm::new(12));
        let mut krylov_phikv_eval = KrylovExpm::new(expmv, m, krylov_dim_max, tol, Some(iom));
        krylov_phikv_eval.set_verbosity(true);

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref()];

        // compute phi_0(dt*A)*b0
        let dt = 0.3;
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi0mv_krylov_pm: Mat<f64> = krylov_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi0mv_pade_dense = matexp(test_b.as_ref(), dt) * test_v.as_ref();
        println!("krylov phi0mv: {:?}", &phi0mv_krylov_pm);
        println!("pade phi0mv: {:?}", &phi0mv_pade_dense);
        mat_mat_approx_eq(phi0mv_krylov_pm.as_ref(), phi0mv_pade_dense.as_ref(), 1e-8);

        // compute phi_1(dt*A)*b0
        let zeros = Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), test_v.as_ref()];
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi1mv_krylov_pm: Mat<f64> = krylov_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi1mv_pade_dense = phi_ext((dt * test_b).as_ref(), 1) * test_v.as_ref();
        println!("krylov phi1mv: {:?}", &phi1mv_krylov_pm);
        println!("pade phi1mv: {:?}", &phi1mv_pade_dense);
        mat_mat_approx_eq(phi1mv_krylov_pm.as_ref(), phi1mv_pade_dense.as_ref(), 1e-8);
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
        let scale = 20.0; // increase stiffness of the problem
        _run_krylov_phikv(faer::Scale(scale) * test_b, test_v);
    }
}
