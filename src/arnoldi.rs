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

/// Scales and orthogonalizes an Arnoldi residual using modified Gram-Schmidt.
///
/// Given the already computed vector $A q_k$, computes
/// $v = \mathrm{scale} \thinspace A q_k$. For each selected basis vector $q_i$,
/// computes $h_{i,k} = v^T q_i$ and updates $v \leftarrow v - h_{i,k} q_i$.
/// Each projection uses the updated residual; `zip!` performs the updates in
/// place without allocating temporary vectors. The scaling is fused into the
/// first projection, so the residual is not traversed separately to scale it.
/// This helper does not apply $A$.
///
/// The incomplete orthogonalization window is
/// `(k + 1).saturating_sub(iom)..=k`, i.e. the `iom` most recent basis vectors
/// (including $q_k$), as in KIOPS and Saad's IOM($iom$). `iom = 2` on a
/// symmetric operator is the Lanczos three-term recurrence. Only the selected
/// entries of `h` are overwritten; the caller must initialize other entries as
/// needed.
/// If `store_next` is true, writes $\Vert v\Vert_2$ to `h[k + 1]` and normalizes
/// the residual, or zeroes it on happy breakdown. Otherwise, leaves the residual
/// unnormalized and does not write the subdiagonal entry.
///
/// # Arguments
///
/// * `residual` - vector $A q_k$, overwritten by the scaled, orthogonalized
///   residual (and normalized or zeroed when `store_next` is true)
/// * `basis` - existing Krylov basis, with the same number of rows as `residual`
///   and at least `k + 1` columns
/// * `h` - Hessenberg column $k$, with at least `k + 1` entries, or `k + 2`
///   entries when `store_next` is true
/// * `scale` - scale factor applied to the input vector
/// * `k` - zero-based index of the current Arnoldi iteration
/// * `iom` - incomplete orthogonalization depth, at least 1; the residual is
///   orthogonalized against the `iom` most recent basis vectors, including the
///   current one
/// * `store_next` - whether to store the subdiagonal norm and prepare the
///   residual as the next basis vector
///
/// # Returns
///
/// True if the residual norm before normalization is below the absolute happy
/// breakdown threshold of `1e-18`; false otherwise.
///
/// # Panics
///
/// Panics if the residual and basis dimensions are incompatible, if basis
/// column `k` is missing, if `h` has insufficient entries, or if `iom` is zero.
fn arnoldi_orthogonalize<T>(
    mut residual: ColMut<T>,
    basis: MatRef<T>,
    mut h: ColMut<T>,
    scale: T,
    k: usize,
    iom: usize,
    store_next: bool,
) -> bool
where
    T: RealField + Float,
{
    // IOM window: the `iom` most recent basis vectors, ending at q_k.
    let first = (k + 1).saturating_sub(iom);
    // Fold the scale into the first projection: with v = A q_k unscaled,
    // h = scale * (v^T q) and v <- scale * v - h q.
    let qi = basis.col(first);
    let ht = scale * (residual.as_ref().transpose() * qi);
    h[first] = ht;
    faer::zip!(residual.rb_mut(), qi).for_each(|faer::unzip!(y, x)| *y = *y * scale - ht * *x);
    for i in first + 1..=k {
        let qi = basis.col(i);
        let ht = residual.as_ref().transpose() * qi;
        h[i] = ht;
        faer::zip!(residual.rb_mut(), qi).for_each(|faer::unzip!(y, x)| *y = *y - ht * *x);
    }
    let norm = residual.norm_l2();
    if store_next {
        h[k + 1] = norm;
    }
    let breakdown = norm < T::from(1e-18).unwrap();
    if store_next && !breakdown {
        let inv_norm = T::from(1.0).unwrap() / norm;
        faer::zip!(residual.rb_mut()).for_each(|faer::unzip!(y)| *y = *y * inv_norm);
    } else if store_next {
        // A consumed terminal column must never contain a stale residual.
        residual.fill(T::from(0.0).unwrap());
    }
    breakdown
}

/// Arnoldi inner iteration with linear operator A
///
/// # Arguments
///
/// * `a_lo` - linear operator, sparse mat or method to apply mat to vec
/// * `a_lo_scale` - scale factor on the linear operator
/// * `k` - current krylov iteration
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth (at least 1)
/// * `hs` - upper hessenberg
/// * `qs` - orthonormal basis of krylov subspace
/// * `extended` - return extended, nonsquare hessenberg
/// * `par` - same parallelism used to size the operator scratch workspace
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
    par: faer::Par,
) -> bool
where
    T: RealField + Float,
{
    let store_next = k + 1 < n || extended;
    if k + 1 < qs.ncols() {
        // Split to borrow the basis and output column without aliasing.
        let (basis, workspace) = qs.rb_mut().split_at_col_mut(k + 1);
        let basis = basis.as_ref();
        let mut residual = workspace.col_mut(0);
        a_lo.apply(
            residual.rb_mut().as_mat_mut(),
            basis.col(k).as_mat(),
            par,
            stack,
        );
        return arnoldi_orthogonalize(
            residual,
            basis,
            hs.col_mut(k),
            a_lo_scale,
            k,
            iom,
            store_next,
        );
    }

    // Last non-extended step: there is no next basis column for workspace.
    let basis = qs.as_ref();
    let mut residual = faer::Col::zeros(qs.nrows());
    a_lo.apply(
        residual.as_mut().as_mat_mut(),
        basis.col(k).as_mat(),
        par,
        stack,
    );
    arnoldi_orthogonalize(
        residual.as_mut(),
        basis,
        hs.col_mut(k),
        a_lo_scale,
        k,
        iom,
        store_next,
    )
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
/// * `iom` - incomplete ortho depth (at least 1; typically 2), i.e. each new
///   vector is orthogonalized against the `iom` most recent basis vectors
///   (including the current one), as in KIOPS
///
/// # Returns
///
/// A tuple `(qs, hs, m)` where `qs` is the orthonormal basis (size
/// `b.nrows()` by `m`), `hs` is the square upper Hessenberg matrix (size `m`
/// by `m`) and `m` is the number of Arnoldi iterations actually performed
/// (the Krylov dimension actually reached, less than the requested value
/// upon happy breakdown).
///
/// # Panics
///
/// Panics if `iom` is zero.
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
    assert!(iom >= 1, "iom must be at least 1");
    let mut breakdown_n = 0;
    let m = std::cmp::min(n, b.nrows());
    let mut hs = faer::Mat::zeros(m, m);
    let mut qs = faer::Mat::zeros(b.nrows(), m);
    if m == 0 {
        return (qs, hs, 0);
    }
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
    let mut mem_buf = MemBuffer::new(a_lo.apply_scratch(1, par));

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
            par,
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
/// On happy breakdown, the terminal basis column is explicitly zeroed, so
/// columns `0..=m` are initialized even when the storage is reused. Do not
/// continue a run after breakdown.
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
/// * `iom` - incomplete ortho depth (at least 1; typically 2), see
///   [`arnoldi_lop`]
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
/// `qs.nrows() != b.nrows()`, or if `qs.ncols() <= i + n`, or if `iom` is
/// zero.
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
    assert!(iom >= 1, "iom must be at least 1");
    // ensure preallocated hessenberg storage is square
    assert!(hs.nrows() == hs.ncols());
    // ensure enough space avail in hs to write into
    assert!(hs.ncols() > i + n);
    assert!(qs.ncols() > i + n);
    // ensure orthonormal matrix has correct number of rows
    assert!(qs.nrows() == dim);
    let max_krylov_dim = hs.ncols();
    let norm_b = b.norm_l2();

    // prevent div by 0 if norm_b~0
    let mut breakdown_n = i;
    let not_early_bkdwn: bool = (T::one() / norm_b).is_finite();
    if i == 0 {
        let scale = if not_early_bkdwn {
            T::from(1.0).unwrap() / norm_b
        } else {
            T::from(1.0).unwrap()
        };
        faer::zip!(qs.rb_mut().col_mut(0), b.col(0)).for_each(|faer::unzip!(q, x)| *q = *x * scale);
    }
    let mut breakdown_flag = !not_early_bkdwn;

    // mem buffer size
    let par = faer::get_global_parallelism();
    let mut mem_buf = MemBuffer::new(a_lo.apply_scratch(1, par));

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
            par,
        );
        breakdown_n += 1;
    }

    (breakdown_flag || breakdown_n >= dim, breakdown_n)
}

#[cfg(test)]
mod test_arnoldi {
    use crate::mat_utils::{dense_to_sprs, mat_mat_approx_eq, random_mat_normal};
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

    // Allocation-heavy MGS retained only as an independent regression oracle.
    fn allocating_reference(
        a: &dyn LinOp<f64>,
        b: MatRef<f64>,
        scale: f64,
        m: usize,
        iom: usize,
    ) -> (Mat<f64>, Mat<f64>) {
        let mut q = Mat::zeros(b.nrows(), m + 1);
        let mut h = Mat::zeros(m + 1, m + 1);
        q.col_mut(0)
            .copy_from((b * faer::Scale(1.0 / b.norm_l2())).col(0));
        let par = faer::get_global_parallelism();
        let mut buf = MemBuffer::new(a.apply_scratch(1, par));
        for k in 0..m {
            let mut v = Mat::zeros(b.nrows(), 1);
            a.apply(v.as_mut(), q.col(k).as_mat(), par, MemStack::new(&mut buf));
            v = v * faer::Scale(scale);
            for i in (k + 1).saturating_sub(iom)..=k {
                let ht = v.col(0).transpose() * q.col(i);
                h[(i, k)] = ht;
                v = v - q.col(i).as_mat() * faer::Scale(ht);
            }
            let norm = v.norm_l2();
            h[(k + 1, k)] = norm;
            if norm < 1e-18 {
                break;
            }
            q.col_mut(k + 1)
                .copy_from((v * faer::Scale(1.0 / norm)).col(0));
        }
        (q, h)
    }

    #[test]
    fn test_arnoldi_in_place_matches_allocating_mgs() {
        let a = Mat::from_fn(20, 20, |i, j| {
            if i == j {
                -((i + 1) as f64)
            } else {
                ((3 * i + 7 * j + 1) as f64).sin() / 20.0
            }
        });
        let b = Mat::from_fn(20, 1, |i, _| ((i + 1) as f64).cos());
        for iom in [1, 2, 1000] {
            let (q_ref, h_ref) = allocating_reference(&a, b.as_ref(), -0.7, 6, iom);
            let mut q = Mat::full(20, 7, f64::NAN);
            let mut h = Mat::zeros(7, 7);
            let (bd, m) =
                arnoldi_lop_restarted(&a, -0.7, b.as_ref(), h.as_mut(), q.as_mut(), 0, 3, iom);
            assert!(!bd);
            assert_eq!(m, 3);
            let (bd, m) =
                arnoldi_lop_restarted(&a, -0.7, b.as_ref(), h.as_mut(), q.as_mut(), 3, 3, iom);
            assert!(!bd);
            assert_eq!(m, 6);
            mat_mat_approx_eq(q.as_ref(), q_ref.as_ref(), 1e-12);
            mat_mat_approx_eq(h.as_ref(), h_ref.as_ref(), 1e-12);
            // Includes the allocated workspace in the last non-extended step.
            let (q_fixed, h_fixed, m) = arnoldi_lop(&a, -0.7, b.as_ref(), 6, iom);
            assert_eq!(m, 6);
            mat_mat_approx_eq(q_fixed.as_ref(), q_ref.get(.., ..6), 1e-12);
            mat_mat_approx_eq(h_fixed.as_ref(), h_ref.get(..6, ..6), 1e-12);
            let aq = faer::Scale(-0.7) * (a.as_ref() * q.get(.., ..6));
            let qh = q.as_ref() * h.get(.., ..6);
            assert!((aq - qh).norm_l2() < 1e-11);
        }
    }

    #[test]
    fn test_arnoldi_breakdown_initializes_terminal_column() {
        let a = Mat::<f64>::zeros(10, 10);
        let b = Mat::full(10, 1, 1.0);
        let mut q = Mat::full(10, 9, f64::NAN);
        let mut h = Mat::zeros(9, 9);
        let (bd, m) = arnoldi_lop_restarted(&a, 1.0, b.as_ref(), h.as_mut(), q.as_mut(), 0, 8, 2);
        assert!(bd);
        assert_eq!(m, 1);
        assert_eq!(q.col(1).norm_l2(), 0.0);
        assert!(q[(0, 2)].is_nan()); // Unconsumed storage need not be touched.
        let zero = Mat::zeros(10, 1);
        q.fill(f64::NAN);
        let (bd, m) =
            arnoldi_lop_restarted(&a, 1.0, zero.as_ref(), h.as_mut(), q.as_mut(), 0, 8, 2);
        assert!(bd);
        assert_eq!(m, 0);
        assert_eq!(q.col(0).norm_l2(), 0.0);
    }

    #[test]
    fn test_arnoldi_iom_window_band() {
        // IOM(iom) orthogonalizes against exactly the `iom` most recent basis
        // vectors, so column k of H is nonzero only in rows k+1-iom..=k+1.
        let a = Mat::from_fn(20, 20, |i, j| {
            if i == j {
                -((i + 1) as f64)
            } else {
                ((3 * i + 7 * j + 1) as f64).sin() / 5.0
            }
        });
        let b = Mat::from_fn(20, 1, |i, _| ((i + 1) as f64).cos());
        let m = 8;
        for iom in [1, 2, 3] {
            let (_, h, bd) = arnoldi_lop(&a, 1.0, b.as_ref(), m, iom);
            assert_eq!(bd, m);
            for k in 0..m {
                for i in 0..m {
                    if i + iom <= k {
                        assert_eq!(h[(i, k)], 0.0, "iom={iom} h[{i},{k}]");
                    } else if i <= k {
                        assert!(h[(i, k)].abs() > 1e-8, "iom={iom} h[{i},{k}]");
                    }
                }
            }
        }
    }

    #[test]
    fn test_arnoldi_iom2_is_lanczos_for_symmetric() {
        // iom=2 is the three-term Lanczos recurrence for a symmetric operator,
        // so it must agree with full orthogonalization.
        let a = Mat::from_fn(20, 20, |i, j| {
            let (lo, hi) = (i.min(j), i.max(j));
            if i == j {
                -((i + 1) as f64)
            } else {
                ((3 * lo + 7 * hi + 1) as f64).sin() / 5.0
            }
        });
        let b = Mat::from_fn(20, 1, |i, _| ((i + 1) as f64).cos());
        let (q2, h2, _) = arnoldi_lop(&a, -0.7, b.as_ref(), 8, 2);
        let (qf, hf, _) = arnoldi_lop(&a, -0.7, b.as_ref(), 8, 1000);
        mat_mat_approx_eq(q2.as_ref(), qf.as_ref(), 1e-9);
        mat_mat_approx_eq(h2.as_ref(), hf.as_ref(), 1e-9);
    }

    #[test]
    #[should_panic(expected = "iom must be at least 1")]
    fn test_arnoldi_lop_rejects_zero_iom() {
        let a = Mat::<f64>::identity(5, 5);
        let b = Mat::full(5, 1, 1.0);
        arnoldi_lop(&a, 1.0, b.as_ref(), 3, 0);
    }

    #[test]
    #[should_panic(expected = "iom must be at least 1")]
    fn test_arnoldi_lop_restarted_rejects_zero_iom() {
        let a = Mat::<f64>::identity(5, 5);
        let b = Mat::full(5, 1, 1.0);
        arnoldi_lop_restarted(
            &a,
            1.0,
            b.as_ref(),
            Mat::zeros(5, 5).as_mut(),
            Mat::zeros(5, 5).as_mut(),
            0,
            3,
            0,
        );
    }

    #[test]
    #[should_panic]
    fn test_arnoldi_restarted_rejects_short_basis() {
        let a = Mat::<f64>::identity(5, 5);
        let b = Mat::full(5, 1, 1.0);
        arnoldi_lop_restarted(
            &a,
            1.0,
            b.as_ref(),
            Mat::zeros(4, 4).as_mut(),
            Mat::zeros(5, 3).as_mut(),
            0,
            3,
            2,
        );
    }

    /// Run alone with --release --ignored --nocapture --test-threads=1.
    #[test]
    #[ignore = "release-mode performance comparison"]
    fn benchmark_arnoldi_workspace_reuse() {
        use std::hint::black_box;
        use std::time::Instant;
        let n = 20_000;
        let m = 40;
        let triplets: Vec<_> = (0..n)
            .map(|i| faer::sparse::Triplet::new(i, i, -(1.0 + i as f64 / n as f64)))
            .collect();
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let b = Mat::from_fn(n, 1, |i, _| ((i + 1) as f64).sin());
        let mut q = Mat::zeros(n, m + 1);
        let mut h = Mat::zeros(m + 1, m + 1);
        let (q_ref, h_ref) = allocating_reference(&a, b.as_ref(), 0.7, m, 2);
        arnoldi_lop_restarted(&a, 0.7, b.as_ref(), h.as_mut(), q.as_mut(), 0, m, 2);
        mat_mat_approx_eq(q.as_ref(), q_ref.as_ref(), 1e-10);
        mat_mat_approx_eq(h.as_ref(), h_ref.as_ref(), 1e-10);
        let mut old_times = Vec::new();
        let mut new_times = Vec::new();
        for round in 0..8 {
            // Alternate order to reduce warm-cache and scheduling bias.
            for old in if round % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let start = Instant::now();
                if old {
                    black_box(allocating_reference(&a, black_box(b.as_ref()), 0.7, m, 2));
                    old_times.push(start.elapsed());
                } else {
                    h.fill(0.0);
                    black_box(arnoldi_lop_restarted(
                        &a,
                        0.7,
                        black_box(b.as_ref()),
                        h.as_mut(),
                        q.as_mut(),
                        0,
                        m,
                        2,
                    ));
                    new_times.push(start.elapsed());
                }
            }
        }
        println!(
            "Arnoldi n={n}, m={m}, iom=2 sum: allocating={:?}, reused={:?}",
            old_times.iter().sum::<std::time::Duration>(), new_times.iter().sum::<std::time::Duration>()
        );
        println!(
            "Arnoldi n={n}, m={m}, iom=2 mean: allocating={:?}, reused={:?}",
            old_times.iter().sum::<std::time::Duration>() / 8, new_times.iter().sum::<std::time::Duration>() / 8
        );
    }

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
