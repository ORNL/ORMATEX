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
//! Arnoldi iteration for building Krylov subspace bases.
//!
//! This module provides the Arnoldi process with optional incomplete
//! orthogonalization for any faer `LinOp` (dense matrices, sparse
//! matrices, or matrix-free operators). Given an operator $A$ and a starting
//! vector $b$, the process builds an orthonormal basis $Q_m$ of the Krylov
//! subspace $$ \mathcal K_m(A, b) = \mathrm{span}\lbrace b, Ab, A^2 b, \ldots, A^{m-1} b\rbrace $$
//! and an upper Hessenberg matrix $H_m = Q_m^T A Q_m$.
//!
//! Key functions:
//!
//! * [`arnoldi_lop`] - allocating Arnoldi iteration.
//! * [`arnoldi_lop_restarted`] - Arnoldi iteration writing into preallocated
//!   storage, which can be continued from a previous iteration.
//!
//! # References
//!
//! * Saad, Y. "Analysis of some Krylov subspace approximations to the matrix
//!   exponential operator", SIAM J. Numer. Anal. 29(1) (1992) 209-228.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer_traits::RealField;
use num_traits::Float;
use reborrow::ReborrowMut;
use std::cmp;
// use faer::dyn_stack::PodStack;

/// Arnoldi inner iteration with linear operator A
///
/// # Arguments
///
/// * `a_lo` - linear operator, sparse mat or method to apply mat to vec
/// * `a_lo_scale` - scale factor on the linear operator
/// * `k` - current krylov iteration
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth
/// * `hs` - upper hessenberg
/// * `qs` - orthonormal basis of krylov subspace
/// * `extended` - return extended, nonsquare hessenberg
///
fn arnoldi_inner_lop<T>(
    a_lo: &dyn LinOp<T>,
    a_lo_scale: T,
    k: usize,
    n: usize,
    iom: usize,
    hs: MatMut<T>,
    mut qs: MatMut<T>,
    stack: &mut MemStack,
    extended: bool,
) -> bool
where
    T: RealField + Float,
{
    // final iter check
    let not_final_it: bool = k + 1 < n;

    // incomplete orth depth
    let iom_depth = cmp::max(k as i32 - iom as i32, 0) as usize;

    // breakdown tol
    let breakdown_tol = T::from(1e-18).unwrap();

    // Krylov vector
    let q_col: ColRef<T> = qs.rb_mut().col(k);

    // let mut qv: Mat<T> = a_lo * q_col;
    let mut qv: Mat<T> = faer::Mat::zeros(q_col.nrows(), 1);
    a_lo.apply(
        qv.as_mut(),
        q_col.as_mat().as_ref(),
        faer::get_global_parallelism(),
        stack,
    );
    qv = qv * faer::Scale(a_lo_scale);

    // let mut h = Vec::with_capacity(k + 2);
    // let mut h = vec![T::from(0.0).unwrap(); k+2];
    let mut h = hs.col_mut(k);
    for i in iom_depth..=k {
        let qci: ColRef<T> = qs.rb_mut().col(i);
        let ht = qv.col(0).transpose() * qci;
        h[i] = ht;
        qv = qv - (qci.as_mat() * faer::Scale(ht));
    }

    let norm_v = qv.norm_l2();
    if k + 1 < n || extended {
        h[k + 1] = norm_v;
    }

    // check for happy breakdown
    let breakdown_flag: bool = norm_v < breakdown_tol;

    if (not_final_it || extended) && !breakdown_flag {
        // if norm_v is zero this is a div by 0 err
        qv = qv * faer::Scale(T::from(1.).unwrap() / norm_v);
        qs.col_mut(k + 1).copy_from(qv.col(0));
    }

    return breakdown_flag;
}

/// Arnoldi iteration with linear operator $A$.
///
/// Builds an orthonormal basis $Q$ of the Krylov subspace
/// $\mathrm{span}\lbrace b, Ab, A^2 b, \ldots\rbrace $ and the corresponding upper
/// Hessenberg matrix $H$ such that $H = Q^T A Q$ (with $A$ scaled by
/// `a_lo_scale`). The starting vector is normalized internally, so the first
/// column of $Q$ is $b / \Vert b\Vert_2$; the caller is responsible for
/// re-applying the factor $\Vert b\Vert_2$ when needed.
///
/// The iteration stops early on happy breakdown, in which case the returned
/// matrices are truncated to the number of iterations performed. The
/// effective Krylov dimension is at most `min(n, b.nrows())`.
/// If $\Vert b\Vert_2$ is so small that its reciprocal is not finite, the vector
/// is not normalized.
///
/// Only the first column of `b` is used to seed the iteration.
///
/// # Arguments
///
/// * `a_lo` - linear operator, sparse mat or method to apply mat to vec
/// * `a_lo_scale` - scale factor on the linear operator
/// * `b` - initial vector $b$ (column) of the Krylov sequence `b, Ab, A^2 b, ...`
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth, i.e. each new vector is orthogonalized
///   against at most the previous `iom` basis vectors
///
/// # Returns
///
/// A tuple `(qs, hs, m)` where `qs` is the orthonormal basis (size
/// `b.nrows()` by `m`), `hs` is the square upper Hessenberg matrix (size `m`
/// by `m`) and `m` is the number of Arnoldi iterations actually performed
/// (the Krylov dimension actually reached, less than the requested value
/// upon happy breakdown).
pub fn arnoldi_lop<T>(
    a_lo: &dyn LinOp<T>,
    a_lo_scale: T,
    b: MatRef<T>,
    n: usize,
    iom: usize,
) -> (Mat<T>, Mat<T>, usize)
where
    T: RealField + Float,
{
    let mut breakdown_n = 0;
    let m = std::cmp::min(n, b.nrows());
    let mut hs = faer::Mat::zeros(m, m);
    let mut qs = faer::Mat::zeros(b.nrows(), m);
    let norm_b = b.norm_l2();

    // prevent div by 0 if norm_b~0
    let not_early_bkdwn: bool = (T::one() / norm_b).is_finite();
    let q0 = if not_early_bkdwn {
        b * faer::Scale(T::from(1.0).unwrap() / norm_b)
    } else {
        b * faer::Scale(T::from(1.0).unwrap())
    };
    qs.col_mut(0).copy_from(q0.col(0));

    // mem buffer size
    let par = faer::get_global_parallelism();
    let mut mem_buf = MemBuffer::new(a_lo.apply_scratch(b.ncols(), par));

    for k in 0..m {
        let breakdown_flag = arnoldi_inner_lop(
            a_lo,
            a_lo_scale,
            k,
            m,
            iom,
            hs.as_mut(),
            qs.as_mut(),
            MemStack::new(&mut mem_buf),
            false,
        );
        breakdown_n += 1;
        if breakdown_flag == true {
            break;
        }
    }

    (
        qs.get(0..b.nrows(), 0..breakdown_n).to_owned(),
        hs.get(0..breakdown_n, 0..breakdown_n).to_owned(),
        breakdown_n,
    )
}

/// Arnoldi iteration that can be restarted, writing into preallocated storage.
///
/// Takes mutable Hessenberg and orthonormal matrices as input and writes
/// into them. This avoids allocating $H$ and $Q$ inside this method, but
/// places the burden of correctly extracting the upper-left $H$ block on the
/// caller.
///
/// This is equal to [`arnoldi_lop`] if `i = 0` and `n` is set to the desired
/// Krylov dimension. To continue a previous run, pass the same `hs` and `qs`
/// with `i` set to the number of iterations already performed.
///
/// Unlike [`arnoldi_lop`], the extended (non-square) Hessenberg entry
/// `h[k+1, k]` is also written in the final iteration, and the corresponding
/// basis vector `q[k+1]` is stored, so `qs` must have at least `i + n + 1`
/// columns.
///
/// # Arguments
///
/// * `a_lo` - linear operator, sparse mat or method to apply mat to vec
/// * `a_lo_scale` - scale factor on the linear operator
/// * `b` - initial vector $b$ (column) of the Krylov sequence `b, Ab, A^2 b, ...`.
///   The normalized $b$ is only written to the first column of `qs` if `i == 0`.
/// * `hs` - hessenberg matrix. View of mutable matrix
/// * `qs` - orthonormal matrix. View of mutable matrix
/// * `i` - index to start from.
/// * `n` - number of additional arnoldi iterations to compute.
/// * `iom` - incomplete ortho depth
///
/// # Returns
///
/// A tuple `(breakdown, m)`. `breakdown` is true if happy breakdown was
/// detected (or the Krylov dimension reached the problem dimension, or $b$ is
/// numerically zero). `m` is the total number of Arnoldi iterations performed
/// so far, i.e. `i` plus the number of iterations run in this call.
///
/// # Panics
///
/// Panics if `hs` is not square, if `hs.ncols() <= i + n`, or if
/// `qs.nrows() != b.nrows()`.
///
pub fn arnoldi_lop_restarted<T>(
    a_lo: &dyn LinOp<T>,
    a_lo_scale: T,
    b: MatRef<T>,
    mut hs: MatMut<T>,
    mut qs: MatMut<T>,
    i: usize,
    n: usize,
    iom: usize,
) -> (bool, usize)
where
    T: RealField + Float,
{
    let dim = b.nrows();
    // ensure preallocated hessenberg storage is square
    assert!(hs.nrows() == hs.ncols());
    // ensure enough space avail in hs to write into
    assert!(hs.ncols() > i + n);
    // ensure orthonormal matrix has correct number of rows
    assert!(qs.nrows() == dim);
    let max_krylov_dim = hs.ncols();
    let norm_b = b.norm_l2();

    // prevent div by 0 if norm_b~0
    let mut breakdown_n = i;
    let not_early_bkdwn: bool = (T::one() / norm_b).is_finite();
    let q0 = if not_early_bkdwn {
        b * faer::Scale(T::from(1.0).unwrap() / norm_b)
    } else {
        b * faer::Scale(T::from(1.0).unwrap())
    };
    if i == 0 {
        qs.rb_mut().col_mut(i).copy_from(q0.col(0));
    }
    let mut breakdown_flag = !not_early_bkdwn;

    // mem buffer size
    let par = faer::get_global_parallelism();
    let mut mem_buf = MemBuffer::new(a_lo.apply_scratch(b.ncols(), par));

    for k in i..i + n {
        // TODO: we should not have to check this.  happy breakdown should
        // happen here
        if k >= dim {
            breakdown_flag = true;
        }
        if breakdown_flag == true {
            break;
        }
        // TODO: check that the last vector is properly computed in
        // the inner loop.
        breakdown_flag = arnoldi_inner_lop(
            a_lo,
            a_lo_scale,
            k,
            max_krylov_dim,
            iom,
            hs.as_mut(),
            qs.as_mut(),
            MemStack::new(&mut mem_buf),
            true,
        );
        breakdown_n += 1;
    }

    (breakdown_flag, breakdown_n)
}

#[cfg(test)]
mod test_arnoldi {
    use crate::mat_utils::{dense_to_sprs, mat_mat_approx_eq, random_mat_normal};
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_arnoldi_lop_dens() {
        // test that arnoldi works with a dense matrix
        let test_a: Mat<f64> = random_mat_normal(10, 10);

        // pick a starting vector and normalize it
        let mut q0: Mat<f64> = random_mat_normal(10, 1);
        q0 = q0.as_ref() * faer::Scale(1.0 / q0.norm_l2());

        // arnoldi with linear op
        let iom = 1000;
        let kd = 10;
        let (q, h, _brkdwn) = arnoldi_lop(&test_a.as_ref(), 1.0, q0.as_ref(), kd, iom);
        println!("arnoldi linop: \n {:?}", q);
        // brkdwn flag < 0 means method terminated without breakdown
        // assert!(_brkdwn < 0);

        // ensure Q is orthonormal
        let qt_q = q.as_ref().transpose() * q.as_ref();
        mat_mat_approx_eq(qt_q.as_ref(), faer::Mat::identity(10, 10).as_ref(), 1.0e-12);

        println!("q shape = {:?}, {:?}", q.nrows(), q.ncols());
        println!("h shape = {:?}, {:?}", h.nrows(), h.ncols());

        // check that Q^T*A*Q = H
        let h_test = (q.as_ref().transpose() * test_a.as_ref() * q.as_ref() - h.as_ref()).norm_l2()
            * (1. / test_a.norm_l2());
        assert_approx_eq!(h_test, 0.0, 1.0e-12);
    }

    #[test]
    fn test_arnoldi_lop_sprs() {
        // test that arnoldi works with a sparse matrix
        let dense_a: Mat<f64> = random_mat_normal(10, 10);
        let test_a = dense_to_sprs(dense_a.as_ref());

        // pick a starting vector
        let q0: Mat<f64> = random_mat_normal(10, 1);
        // q0 = q0.as_ref() * faer::Scale(1.0 / q0.norm_l2());

        // arnoldi with linear op
        let iom = 1000;
        let kd = 10;
        let (q, h, _brkdwn) = arnoldi_lop(&test_a.as_ref(), 1.0, q0.as_ref(), kd, iom);
        println!("arnoldi linop: \n {:?}", q);
        // brkdwn flag < 0 means method terminated without breakdown
        // assert!(_brkdwn < 0);

        // ensure Q is orthonormal
        let qt_q = q.as_ref().transpose() * q.as_ref();
        mat_mat_approx_eq(qt_q.as_ref(), faer::Mat::identity(10, 10).as_ref(), 1.0e-12);

        println!("q shape = {:?}, {:?}", q.nrows(), q.ncols());
        println!("h shape = {:?}, {:?}", h.nrows(), h.ncols());

        // check that Q^T*A*Q = H
        let h_test = (q.as_ref().transpose() * test_a.as_ref() * q.as_ref() - h.as_ref()).norm_l2()
            * (1. / test_a.to_dense().norm_l2());
        assert_approx_eq!(h_test, 0.0, 1.0e-12);
    }

    #[test]
    fn test_arnoldi_lop_restarted() {
        use crate::mat_utils::mat_mat_approx_eq;
        // Test ability to restart the arnoldi procedure to continue
        // from where we left off.
        let dim_a = 10;
        let dense_a: Mat<f64> = random_mat_normal(dim_a, dim_a);

        // pick a starting vector
        let q0: Mat<f64> = random_mat_normal(dim_a, 1);

        // run restarted arnoldi for 6 iterations
        let m = 6;
        let mut hs_full = faer::Mat::zeros(dim_a, dim_a);
        let mut qs_full = faer::Mat::zeros(dim_a, dim_a);
        let iom = 10;
        let (bkdwn, _) = arnoldi_lop_restarted(
            &dense_a,
            1.0,
            q0.as_ref(),
            hs_full.as_mut(),
            qs_full.as_mut(),
            0,
            m,
            iom,
        );
        assert!(!bkdwn);

        // run the standard allocating arnoldi procedure for 6 iterations
        let (q, h, _) = arnoldi_lop(&dense_a, 1.0, q0.as_ref(), m, iom);

        // Check output is equal to existing allocating arnoldi procedure.
        let hs_slice = hs_full.get(0..m, 0..m);
        mat_mat_approx_eq(h.as_ref(), hs_slice, f64::EPSILON * 100.);

        let qs_slice = qs_full.get(.., 0..m);
        mat_mat_approx_eq(q.as_ref(), qs_slice, f64::EPSILON * 100.);

        // continue two more iterations, for a total of 8
        let (bkdwn, _) = arnoldi_lop_restarted(
            &dense_a,
            1.0,
            q0.as_ref(),
            hs_full.as_mut(),
            qs_full.as_mut(),
            m,
            2,
            iom,
        );
        assert!(!bkdwn);
        // run the standard allocating arnoldi procedure for 8 iterations
        let (q, h, _) = arnoldi_lop(&dense_a, 1.0, q0.as_ref(), m + 2, iom);

        // Check output is equal to existing allocating arnoldi procedure.
        let hs_slice = hs_full.get(0..m + 2, 0..m + 2);
        mat_mat_approx_eq(h.as_ref(), hs_slice, f64::EPSILON * 100.);

        let qs_slice = qs_full.get(.., 0..m + 2);
        mat_mat_approx_eq(q.as_ref(), qs_slice, f64::EPSILON * 100.);

        // visual check with: cargo test cargo test lop_restarted -- --nocapture
        println!("arnoldi_lop_restarted h: {:?}", hs_full.as_ref());
        println!("arnoldi_lop h: {:?}", h.as_ref());
    }
}
