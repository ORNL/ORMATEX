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
use crate::mat_utils::scaled_mm;
use faer::linalg::matmul::matmul;
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::{Accum, Par};
use std::cmp::{max, min};
use std::collections::HashMap;

/// Krylov methods to compute the sparse (matrix-free) matrix exponential
/// and phi-function vector products.
///
/// Holds the Krylov dimension state (current, maximum, adaptive increment and
/// lookback), the incomplete orthogonalization depth, the tolerance, the
/// number of substeps, and work storage for the Arnoldi basis and Hessenberg
/// matrix. The adaptive evaluation updates the current Krylov dimension, so it
/// persists between calls.
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
    /// number of equal substeps used to evaluate one phi-function product,
    /// 0 disables substepping
    max_substeps: usize,
    /// last adapted krylov dim for each kind of adaptive call, keyed by
    /// (phi order, vector length). Used as the starting dimension of the next
    /// call of the same kind.
    m_by_call: HashMap<(usize, usize), usize>,
    /// parallelism for the dense phi-function of the small Hessenberg matrix
    dense_par: Par,
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
    /// to 10, the number of substeps to 0 (no substepping), and verbosity is
    /// off. See [`KrylovExpm::set_krylov_dim_inc`],
    /// [`KrylovExpm::set_krylov_dim_lookback`], [`KrylovExpm::set_max_substeps`]
    /// and [`KrylovExpm::set_verbosity`].
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
            dense_par: Par::Seq,
            hs: faer::Mat::zeros(0, 0),
            qs: faer::Mat::zeros(0, 0),
            iom: iom_in.unwrap_or(2),
            max_substeps: 0,
            m_by_call: HashMap::new(),
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

    /// Sets the parallelism used by the dense evaluator for the phi-function of
    /// the small Hessenberg matrix.
    ///
    /// Defaults to `Par::Seq`, which is usually best since the Hessenberg
    /// matrix is small (the Krylov dimension, typically a few hundred at most).
    /// This does not affect the product of the Krylov basis with the
    /// phi-function, which is tall and skinny and uses faer's global
    /// parallelism.
    ///
    /// # Arguments
    ///
    /// * `par` - faer parallelism for the dense evaluator
    pub fn set_dense_par(&mut self, par: Par) {
        self.dense_par = par;
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

    /// Sets the number of substeps.
    ///
    /// With `max_substeps = N > 0`, [`KrylovExpm::apply_linop_ext`] (and so
    /// `apply_phi_k_v`) and [`KrylovExpm::apply_phik_linop`] (and so
    /// `apply_phi_k` and [`KrylovExpm::apply_linop`]) evaluate the product in
    /// $N$ equal substeps of size $dt / N$, using the same tolerance in each
    /// substep. The total error therefore grows roughly like $N$ times the
    /// per-substep tolerance. This is a fixed substep count, as in the Leja
    /// evaluator. [`KrylovExpm::apply_phik_linop_adapt`] and
    /// [`KrylovExpm::apply_phik_linop_fixed`] are not substepped.
    ///
    /// # Arguments
    ///
    /// * `max_substeps` - number of substeps. 0 disables substepping
    pub fn set_max_substeps(&mut self, max_substeps: usize) {
        self.max_substeps = max_substeps;
    }

    /// Returns the number of substeps. 0 means substepping is disabled.
    pub fn max_substeps(&self) -> usize {
        self.max_substeps
    }

    /// Sets the current Krylov subspace dimension.
    ///
    /// This is the dimension used by [`KrylovExpm::apply_phik_linop_fixed`]. The
    /// adaptive methods only use it as the starting dimension: they update it
    /// after each converged evaluation, so it rarely needs to be set by hand
    /// unless the fixed dimension method is used.
    ///
    /// This also clears the starting dimensions the adaptive methods remember
    /// for each kind of call (see [`KrylovExpm::apply_phik_linop_adapt`]), so the
    /// next adaptive call of every kind starts from `m`.
    ///
    /// # Arguments
    ///
    /// * `m` - Krylov subspace dimension
    ///
    /// # Panics
    ///
    /// Panics if `m` is zero or greater than the maximum Krylov dimension.
    pub fn set_m(&mut self, m: usize) {
        assert!(
            m >= 1 && m <= self.krylov_dim_max,
            "m must be in 1..=krylov_dim_max"
        );
        self.m = m;
        self.m_by_call.clear();
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
    /// The final dimension is remembered separately for each kind of call,
    /// identified by the phi order `k` and the length of `v0`, and a later call
    /// of the same kind starts from it. For example the two stages of an
    /// exponential Rosenbrock step, which need different dimensions, each keep
    /// their own. A call of a kind not seen before starts from the current
    /// dimension (see [`KrylovExpm::set_m`], which also forgets the remembered
    /// dimensions).
    ///
    /// This method mutates the stored Krylov dimension and work storage. With
    /// verbosity on (see [`KrylovExpm::set_verbosity`]) it prints a summary
    /// (convergence, dimension, error estimate and timing) to stdout.
    ///
    /// The product is not substepped; see [`KrylovExpm::apply_phik_linop`] for
    /// the substepped evaluation.
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
        if self.verbose {
            println!("=== Adaptive KrylovExpm");
        }
        // Allocate storage matrices with correct dimensions
        // The storage must be large enough to hold:
        // - hs: square matrix at least (m+1) x (m+1) where m can grow up to krylov_dim_max
        // - qs: (v0.nrows()) x (m+1) where m can grow up to krylov_dim_max
        // The basis storage only grows in rows, so alternating between operators
        // of different sizes (e.g. an extended operator and its inner operator)
        // does not reallocate. Arnoldi works on the leading `v0_dim` rows.
        let v0_dim = v0.nrows();
        // Start from the dimension this kind of call (phi order, vector length)
        // converged to last time, so that calls of different kinds that share
        // this evaluator (e.g. the stages of an exponential Rosenbrock method)
        // do not disturb each other's starting dimension.
        let call_key = (k, v0_dim);
        if let Some(&m_call) = self.m_by_call.get(&call_key) {
            self.m = m_call;
        }
        let storage_size = self.krylov_dim_max + 1; // +1 for extended Hessenberg
        if self.hs.nrows() != storage_size || self.hs.ncols() != storage_size {
            self.hs = faer::Mat::zeros(storage_size, storage_size);
        }
        if self.qs.nrows() < v0_dim || self.qs.ncols() != storage_size {
            self.qs = faer::Mat::zeros(v0_dim, storage_size);
        }

        // IOM leaves some Hessenberg entries unwritten. Arnoldi overwrites the
        // consumed basis prefix, including the terminal column on breakdown.
        self.hs.fill(0.0);

        // run initial arnoldi iterations up to the current krylov dim
        let (mut breakdown_flag, mut breakdown_m) = arnoldi_lop_restarted(
            a_lo,
            dt,
            v0,
            self.hs.as_mut(),
            self.qs.get_mut(0..v0_dim, ..),
            0,
            self.m,
            self.iom,
        );
        let mut qs_initialized = breakdown_m + 1;

        const BUFFER_M: usize = 2;
        let beta = v0.norm_l2();
        let mut res = v0.to_owned();
        let mut converged = false;
        let mut adapt_iter = 1;
        let mut err_est_p = 0.0;
        while !converged {
            // trim hessenberg to size
            let h_dim = min(self.m, breakdown_m);
            // Match freshly zeroed storage if the legacy control flow includes
            // columns skipped by Arnoldi (e.g. after a dimension-limit exit).
            if qs_initialized < h_dim + 1 {
                self.qs
                    .get_mut(0..v0_dim, qs_initialized..h_dim + 1)
                    .fill(0.0);
                qs_initialized = h_dim + 1;
            }
            // get H_{m+1} view
            let h = self.hs.get(0..h_dim + 1, 0..h_dim + 1);
            let q = self.qs.get(0..v0_dim, 0..h_dim + 1);

            // compute the dense matrix exponential of the hessenberg
            let mut unit_vec = faer::Mat::zeros(h.nrows(), 1);
            unit_vec[(0, 0)] = 1.0;
            let phi_h = self
                .expmv
                .apply_phi_k(h.as_ref(), 1.0, unit_vec.as_ref(), k, self.dense_par);
            res = scaled_mm(beta, q.as_ref(), phi_h.as_ref(), faer::get_global_parallelism());

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
                    println!(
                        "i: {adapt_iter}, m: {last_m}, mp: {last_m_p}, e_mp: {:.5e}, e_m: {:.5e} conv: {converged}, bdwn: {breakdown_flag}",
                        err_est_p, err_est_m
                    );
                }
                log::info!(
                    "adapt i: {adapt_iter}, mp: {last_m_p}, err: {:.6e}, converged: {converged}, bkdwn: {breakdown_flag}",
                    err_est_p
                );

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
                    // Preserve the original requested restart index, but never
                    // pass a stale seed or gap to an operator in reused storage.
                    if qs_initialized < self.m + 1 {
                        self.qs
                            .get_mut(0..v0_dim, qs_initialized..self.m + 1)
                            .fill(0.0);
                    }
                    let (bd, bd_n) = arnoldi_lop_restarted(
                        a_lo,
                        dt,
                        v0,
                        self.hs.as_mut(),
                        self.qs.get_mut(0..v0_dim, ..),
                        self.m,
                        dim_inc,
                        self.iom,
                    );
                    qs_initialized = bd_n + 1;
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

        if self.verbose {
            println!(
                "converged: {converged}, m: {}, err_est: {:0.6e}",
                self.m, err_est_p
            );
            println!("Krylov expmv time (s): {}", clock.elapsed().as_secs_f64());
        }
        // remember the dimension for the next call of this kind
        self.m_by_call.insert(call_key, self.m);
        // return final approximation beta*Q*exp(H)*e1
        res
    }

    /// Computes $\varphi_k(A \thinspace dt) v_0$ where `A` is a linear operator,
    /// optionally in substeps.
    ///
    /// This is the method used by the
    /// [`crate::matexp_traits::LinOpPhikvEvaluator`] implementation. The Krylov
    /// dimension is adapted as in [`KrylovExpm::apply_phik_linop_adapt`].
    ///
    /// If `max_substeps` is 0 (the default) this is
    /// [`KrylovExpm::apply_phik_linop_adapt`]. If it is $N > 0$ (see
    /// [`KrylovExpm::set_max_substeps`]) the product is evaluated in $N$ equal
    /// substeps. Since $\varphi_k$ for $k \ge 1$ has no semigroup property, the
    /// substeps are applied to the exponential of the extended operator built
    /// from $A \thinspace dt$ and $v_0$ (placed at index $k$ of the right hand
    /// sides), and the full extended vector is carried from one substep to the
    /// next. This is the same construction the Leja evaluator uses. In this case
    /// `v0` must be a single column.
    ///
    /// Use [`KrylovExpm::apply_phik_linop_fixed`] for a fixed Krylov dimension.
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
        &mut self,
        a_lo: &dyn LinOp<f64>,
        dt: f64,
        v0: MatRef<f64>,
        k: usize,
    ) -> Mat<f64> {
        if self.max_substeps == 0 {
            return self.apply_phik_linop_adapt(a_lo, dt, v0, k);
        }
        // phi_k(dt*A) v0 is the first n rows of exp(ext_a_lo) [v0; 0..0; 1]
        // with the extended operator [[dt*A, B], [0, K]].
        let zeros = Mat::<f64>::zeros(v0.nrows(), v0.ncols());
        let mut vbk: Vec<MatRef<f64>> = (0..k).map(|_| zeros.as_ref()).collect();
        vbk.push(v0);
        let ext_a_lo = DynRefExtendedLinOp::new(dt, a_lo, &vbk);
        let (w, n) = ext_a_lo.get_v(&vbk);
        let w = self.adapt_substepped(&ext_a_lo, 1.0, w);
        w.get(0..n, 0..1).to_owned()
    }

    /// Computes $\varphi_k(A \thinspace dt) v_0$ with a fixed Krylov dimension.
    ///
    /// Uses the current Krylov dimension `m` (see [`KrylovExpm::set_m`]) and no
    /// error control, so the caller is responsible for choosing a dimension
    /// large enough for the requested accuracy. The Arnoldi iteration is run on
    /// $A$ and the time step `dt` is applied in the dense evaluation of the
    /// Hessenberg matrix phi-function. Unlike the adaptive methods this does not
    /// modify the stored Krylov dimension, is not substepped, and allocates its
    /// own Krylov basis.
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
    pub fn apply_phik_linop_fixed(
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
        let phi_v = self
            .expmv
            .apply_phi_k(h.as_ref(), dt, unit_vec.as_ref(), k, self.dense_par);
        return scaled_mm(beta, q.as_ref(), phi_v.as_ref(), faer::get_global_parallelism());
    }

    /// Computes $\exp(\tau A_{ext}) w_0$ in `max(max_substeps, 1)` equal substeps
    /// with the adaptive Krylov dimension, and returns the full extended vector.
    ///
    /// With `max_substeps = 0` this is a single call to
    /// [`KrylovExpm::apply_phik_linop_adapt`]. Since
    /// $\exp(\tau A) = \exp(\tau A / N)^N$, the full extended vector is carried
    /// from one substep to the next.
    fn adapt_substepped(
        &mut self,
        ext_a_lo: &DynRefExtendedLinOp,
        tau: f64,
        w0: Mat<f64>,
    ) -> Mat<f64> {
        let n_sub = max(self.max_substeps, 1);
        let tau_sub = tau / n_sub as f64;
        let mut w = w0;
        for _ in 0..n_sub {
            w = self.apply_phik_linop_adapt(ext_a_lo, tau_sub, w.as_ref(), 0);
        }
        w
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
    ///
    /// If `max_substeps` is $N > 0$ (see [`KrylovExpm::set_max_substeps`]) the
    /// exponential is evaluated in $N$ equal substeps of size `tau / N`, each
    /// with the same tolerance, and the full extended vector is carried from
    /// one substep to the next. The adaptive Krylov dimension is then chosen
    /// per substep.
    /// TODO: Implement adaptive selection of the number of substeps.
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
        let (w, n) = ext_a_lo.get_v(vb);

        // compute phi_0(tau*A_ext)*v_ext with adaptive krylov dimension
        let w = self.adapt_substepped(ext_a_lo, tau, w);

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

    fn apply_phi_k(
        &mut self,
        a_lo: &dyn LinOp<f64>,
        dt: f64,
        v: MatRef<f64>,
        k: usize,
    ) -> Mat<f64> {
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

    fn _run_krylov_phikv(test_b: Mat<f64>, test_v: Mat<f64>, iom: usize) {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // krylov method.
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
        let phi0mv_pade_dense = matexp(test_b.as_ref(), dt, Par::Seq) * test_v.as_ref();
        println!("krylov phi0mv: {:?}", &phi0mv_krylov_pm);
        println!("pade phi0mv: {:?}", &phi0mv_pade_dense);
        mat_mat_approx_eq(phi0mv_krylov_pm.as_ref(), phi0mv_pade_dense.as_ref(), 1e-8);

        // compute phi_1(dt*A)*b0
        let zeros = Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), test_v.as_ref()];
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi1mv_krylov_pm: Mat<f64> = krylov_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi1mv_pade_dense = phi_ext((dt * test_b).as_ref(), 1, Par::Seq) * test_v.as_ref();
        println!("krylov phi1mv: {:?}", &phi1mv_krylov_pm);
        println!("pade phi1mv: {:?}", &phi1mv_pade_dense);
        mat_mat_approx_eq(phi1mv_krylov_pm.as_ref(), phi1mv_pade_dense.as_ref(), 1e-8);
    }

    #[test]
    fn test_krylov_phikv_small() {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // krylov method for a small 3x3 A
        let (test_b, test_v) = gen_test_b();
        // The system is smaller than the Krylov dimension and nonsymmetric,
        // so exactness requires full orthogonalization (iom=2 is truncated).
        _run_krylov_phikv(test_b, test_v, 10);
    }

    #[test]
    fn test_krylov_phikv_large() {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // krylov method for a larger 80x80 A
        let (test_b, test_v) = gen_test_c(80);
        let scale = 20.0; // increase stiffness of the problem
        _run_krylov_phikv(faer::Scale(scale) * test_b, test_v, 2);
    }

    fn new_eval(m: usize, krylov_dim_max: usize, iom: usize, max_substeps: usize) -> KrylovExpm {
        let expmv = Box::new(crate::matexp_pade::PadeExpm::new(12));
        let mut eval = KrylovExpm::new(expmv, m, krylov_dim_max, 1e-12, Some(iom));
        eval.set_max_substeps(max_substeps);
        eval
    }

    fn max_abs_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
        (a - b).norm_max()
    }

    #[test]
    fn test_krylov_max_substeps_default_and_setter() {
        let expmv = Box::new(crate::matexp_pade::PadeExpm::new(12));
        let mut eval = KrylovExpm::new(expmv, 10, 20, 1e-8, None);
        assert_eq!(eval.max_substeps, 0);
        eval.set_max_substeps(4);
        assert_eq!(eval.max_substeps, 4);
        assert_eq!(eval.max_substeps(), 4);
        eval.set_max_substeps(0);
        assert_eq!(eval.max_substeps, 0);
    }

    /// Adaptive `apply_phi_k_v`: substepped results agree with the unsubstepped
    /// result and with the dense Pade reference for several right hand sides.
    /// The mixed case exercises the augmented block carried between substeps.
    fn _run_krylov_substep_phi_k_v(
        test_b: Mat<f64>,
        test_v: Mat<f64>,
        dt: f64,
        iom: usize,
        tol_cmp: f64,
    ) {
        let n = test_v.nrows();
        let v1 = Mat::from_fn(n, 1, |i, _| ((i + 1) as f64).sin());
        let v2 = Mat::from_fn(n, 1, |i, _| 0.5 * ((2 * i + 1) as f64).cos());
        let zeros = Mat::<f64>::zeros(n, 1);
        let cases: Vec<Vec<MatRef<f64>>> = vec![
            vec![test_v.as_ref()],
            vec![zeros.as_ref(), test_v.as_ref()],
            vec![zeros.as_ref(), zeros.as_ref(), test_v.as_ref()],
            vec![test_v.as_ref(), v1.as_ref(), v2.as_ref()],
        ];
        for vb in &cases {
            let mut reference = Mat::<f64>::zeros(n, 1);
            for (j, v) in vb.iter().enumerate() {
                let z = faer::Scale(dt) * test_b.as_ref();
                reference += phi_ext(z.as_ref(), j, Par::Seq) * v.as_ref();
            }
            let eval = |max_substeps: usize| {
                let mut e = new_eval(10, 100, iom, max_substeps);
                let ext_lo = DynRefExtendedLinOp::new(dt, &test_b, vb);
                e.apply_phi_k_v(&ext_lo, 1.0, vb)
            };
            let base = eval(0);
            println!(
                "p={} base vs pade: {:.3e}",
                vb.len() - 1,
                max_abs_diff(&base, &reference)
            );
            mat_mat_approx_eq(base.as_ref(), reference.as_ref(), tol_cmp);
            for max_substeps in [2, 4] {
                let sub = eval(max_substeps);
                println!(
                    "p={} substeps={max_substeps} vs base: {:.3e}, vs pade: {:.3e}",
                    vb.len() - 1,
                    max_abs_diff(&sub, &base),
                    max_abs_diff(&sub, &reference)
                );
                mat_mat_approx_eq(sub.as_ref(), base.as_ref(), tol_cmp);
                mat_mat_approx_eq(sub.as_ref(), reference.as_ref(), tol_cmp);
            }
        }
    }

    /// Adaptive `apply_phi_k` (and the `apply_linop` alias): substepped
    /// results agree with the unsubstepped result and the dense Pade reference.
    fn _run_krylov_substep_phi_k(
        test_b: Mat<f64>,
        test_v: Mat<f64>,
        dt: f64,
        iom: usize,
        tol_cmp: f64,
    ) {
        for k in 0..=3 {
            let z = faer::Scale(dt) * test_b.as_ref();
            let reference = phi_ext(z.as_ref(), k, Par::Seq) * test_v.as_ref();
            let eval = |max_substeps: usize| {
                new_eval(10, 100, iom, max_substeps).apply_phi_k(&test_b, dt, test_v.as_ref(), k)
            };
            let base = eval(0);
            println!(
                "k={k} base vs pade: {:.3e}",
                max_abs_diff(&base, &reference)
            );
            mat_mat_approx_eq(base.as_ref(), reference.as_ref(), tol_cmp);
            for max_substeps in [2, 4] {
                let sub = eval(max_substeps);
                println!(
                    "k={k} substeps={max_substeps} vs base: {:.3e}, vs pade: {:.3e}",
                    max_abs_diff(&sub, &base),
                    max_abs_diff(&sub, &reference)
                );
                mat_mat_approx_eq(sub.as_ref(), base.as_ref(), tol_cmp);
                mat_mat_approx_eq(sub.as_ref(), reference.as_ref(), tol_cmp);
            }
        }
        // the exponential alias is substepped as well
        let reference = matexp(test_b.as_ref(), dt, Par::Seq) * test_v.as_ref();
        let sub = new_eval(10, 100, iom, 4).apply_linop(&test_b, dt, test_v.as_ref());
        mat_mat_approx_eq(sub.as_ref(), reference.as_ref(), tol_cmp);
    }

    #[test]
    fn test_krylov_substep_phi_k_v_small() {
        let (test_b, test_v) = gen_test_b();
        _run_krylov_substep_phi_k_v(test_b, test_v, 0.3, 10, 1e-12);
    }

    #[test]
    fn test_krylov_substep_phi_k_v_large() {
        let (test_b, test_v) = gen_test_c(80);
        _run_krylov_substep_phi_k_v(faer::Scale(20.0) * test_b, test_v, 0.3, 2, 1e-10);
    }

    #[test]
    fn test_krylov_substep_phi_k_small() {
        let (test_b, test_v) = gen_test_b();
        _run_krylov_substep_phi_k(test_b, test_v, 0.3, 10, 1e-12);
    }

    #[test]
    fn test_krylov_substep_phi_k_large() {
        let (test_b, test_v) = gen_test_c(80);
        _run_krylov_substep_phi_k(faer::Scale(20.0) * test_b, test_v, 0.3, 2, 1e-10);
    }

    #[test]
    fn test_krylov_substep_helps_convergence() {
        // With a maximum Krylov dimension that is too small for the full
        // (stiff) step, the adaptive method cannot reach the tolerance, but
        // substepping does.
        let (test_b, test_v) = gen_test_c(80);
        let test_b = faer::Scale(20.0) * test_b;
        let dt = 0.3;
        let reference = matexp(test_b.as_ref(), dt, Par::Seq) * test_v.as_ref();
        let kmax = 12;
        let err = |max_substeps: usize| {
            let res = new_eval(kmax, kmax, kmax, max_substeps).apply_phi_k(
                &test_b,
                dt,
                test_v.as_ref(),
                0,
            );
            max_abs_diff(&res, &reference)
        };
        let (err0, err8) = (err(0), err(8));
        println!("kmax={kmax}: N=0 error {err0:.3e}, N=8 error {err8:.3e}");
        // observed: about 4e-4 and 1e-13
        assert!(err0 > 1e-6);
        assert!(err8 < 1e-10);
    }

    #[test]
    fn test_krylov_m_remembered_per_call_kind() {
        // The adaptive methods remember the converged dimension separately for
        // each kind of call (phi order, vector length), and start from it.
        let (test_b, test_v) = gen_test_c(80);
        // a mildly stiff system, for which a small dimension suffices
        let dt = 0.3;
        let n = test_v.nrows();
        let v1 = Mat::from_fn(n, 1, |i, _| ((i + 1) as f64).sin());
        let vb = vec![test_v.as_ref(), v1.as_ref()];
        let ext_lo = DynRefExtendedLinOp::new(dt, &test_b, &vb);
        let key_ext = (0, n + 1);
        let key_phi3 = (3, n);

        // alternating call kinds do not overwrite each other's dimension
        let mut eval = new_eval(10, 100, 2, 0);
        assert!(eval.m_by_call.is_empty());
        for _ in 0..3 {
            eval.apply_phi_k_v(&ext_lo, 1.0, &vb);
            let m_ext = eval.m;
            assert_eq!(eval.m_by_call[&key_ext], m_ext);
            eval.apply_phi_k(&test_b, dt, test_v.as_ref(), 3);
            assert_eq!(eval.m_by_call[&key_phi3], eval.m);
            assert_eq!(eval.m_by_call[&key_ext], m_ext);
        }
        assert_eq!(eval.m_by_call.len(), 2);
        // phi_3 converges faster than phi_0 so the two kinds settle apart
        assert!(eval.m_by_call[&key_ext] > eval.m_by_call[&key_phi3]);

        // a remembered dimension is the starting dimension of the next call of
        // its kind: an over-provisioned start only shrinks by the lookback
        // (here from 60), while a fresh evaluator starting from 12 settles
        // near the dimension actually needed
        let mut remembered = new_eval(12, 100, 2, 0);
        remembered.m_by_call.insert(key_ext, 60);
        remembered.apply_phi_k_v(&ext_lo, 1.0, &vb);
        let mut fresh = new_eval(12, 100, 2, 0);
        fresh.apply_phi_k_v(&ext_lo, 1.0, &vb);
        // observed: 53 (= 60 - 7) and 25
        assert!(remembered.m > 40);
        assert!(fresh.m < 40);

        // set_m forgets the remembered dimensions
        eval.set_m(10);
        assert_eq!(eval.m, 10);
        assert!(eval.m_by_call.is_empty());
    }

    #[test]
    fn test_krylov_fixed_dimension() {
        // The fixed dimension method uses the user chosen dimension, is not
        // substepped and does not modify the stored dimension.
        let (test_b, test_v) = gen_test_c(80);
        let test_b = faer::Scale(20.0) * test_b;
        let dt = 0.3;
        // full orthogonalization and a dimension spanning the whole space
        let mut eval = new_eval(10, 100, 90, 0);
        eval.set_m(90);
        assert_eq!(eval.m, 90);
        let mut base = vec![];
        for k in 0..=3 {
            let z = faer::Scale(dt) * test_b.as_ref();
            let reference = phi_ext(z.as_ref(), k, Par::Seq) * test_v.as_ref();
            let res = eval.apply_phik_linop_fixed(&test_b, dt, test_v.as_ref(), k);
            println!(
                "fixed k={k} vs pade: {:.3e}",
                max_abs_diff(&res, &reference)
            );
            mat_mat_approx_eq(res.as_ref(), reference.as_ref(), 1e-10);
            base.push(res);
        }
        assert_eq!(eval.m, 90);
        // not substepped: identical result with substeps enabled
        eval.set_max_substeps(4);
        for k in 0..=3 {
            let res = eval.apply_phik_linop_fixed(&test_b, dt, test_v.as_ref(), k);
            assert_eq!(max_abs_diff(&res, &base[k]), 0.0);
        }
    }
}
