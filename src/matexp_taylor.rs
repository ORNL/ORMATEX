/*
 * Copyright© 2025,2026 UT-Battelle, LLC
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
/// Taylor series matrix exponential evaluation methods for dense faer Mats.
///
use faer::prelude::*;
use faer::linalg::matmul;
use faer_traits::math_utils::{add, mul};
use faer::traits::ComplexField;
use num_traits::ToPrimitive;
use faer_traits::math_utils::from_f64;
use statrs::function::factorial;
use crate::matexp_traits::DensePhikvEvaluator;

/// Compute the dense matrix exponential using the taylor series.
///
/// # Args
/// * `A` : the matrix
/// * `p` : taylor polynomial order
///
fn matexp_ts<T: ComplexField>(
    a: MatRef<T>,
    p: usize,
) -> Mat<T> {
    let mut m: Mat<T> = a.to_owned();
    let mut m_next = Mat::<T>::zeros(m.nrows(), m.ncols());
    let mut ts_expm: Mat<T> = faer::Mat::identity(m.nrows(), m.ncols());
    let mut fact = factorial::factorial(0 as u64);
    ts_expm = ts_expm / fact;
    for i in 0..p {
        fact *= (i + 1) as f64;
        ts_expm += m.as_ref() / fact;
        matmul::matmul(
            m_next.as_mut(),
            faer::Accum::Replace,
            a,
            m.as_ref(),
            from_f64::<T>(1.0),
            faer::get_global_parallelism(),
        );
        std::mem::swap(&mut m, &mut m_next);
    }
    ts_expm
}

/// Compute the dense matrix exponential using the taylor series with
/// scaling and squaring.
///
/// # Args
/// * `A` : the matrix
/// * `p` : taylor polynomial order
///
fn matexp_ts_ss<T>(
    a: MatRef<T>,
    p: usize,
) -> Mat<T>
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    // compute scaling factor (in powers of 2)
    let s_scale = a
        .norm_max()
        .to_f64()
        .expect("T::Real must be convertible to f64 for Taylor scaling");
    let s = if s_scale > 1.0 {
        s_scale.log2().ceil() as usize
    } else {
        0
    };
    let hs = 1.0 / (2.0_f64).powi(s as i32);

    // compute exp(hs*a)
    let mut matexp_a = matexp_ts((hs * a).as_ref(), p);
    let mut matexp_a_next = Mat::<T>::zeros(matexp_a.nrows(), matexp_a.ncols());

    for _ in 0..s {
        matmul::matmul(
            matexp_a_next.as_mut(),
            faer::Accum::Replace,
            matexp_a.as_ref(),
            matexp_a.as_ref(),
            from_f64::<T>(1.0),
            faer::get_global_parallelism(),
        );
        std::mem::swap(&mut matexp_a, &mut matexp_a_next);
    }
    matexp_a
}

/// Computes the phi_k function using the taylor series with scaling and squaring
/// and the extension formula.
///
/// # Args
/// * `A` : the matrix
/// * `k` : phi-fn order
///
pub fn phik_taylor_ext<T>(z: MatRef<T>, k: usize) -> Mat<T>
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    let n = z.nrows();
    let m = z.ncols();
    assert_eq!(n, m, "phik_taylor_ext requires a square matrix");

    let z_ext: Mat<T> = match k {
        0 => z.to_owned(),
        _ => {
            let z_ext_k_nrows = n + (k - 1) * n;
            let z_ext_k_ncols = m;
            let z_ext_nrows = z_ext_k_nrows + n;
            let z_ext_ncols = z_ext_k_ncols + k * n;
            let mut z_ext = Mat::<T>::zeros(z_ext_nrows, z_ext_ncols);
            z_ext.get_mut(0..n, 0..m).copy_from(z);
            z_ext
                .get_mut(0..z_ext_k_nrows, z_ext_k_ncols..)
                .copy_from(Mat::<T>::identity(k * n, k * n));
            z_ext
        }
    };
    let phi_ks = matexp_ts_ss(z_ext.as_ref(), 16);
    phi_ks.get(0..n, phi_ks.ncols() - n..).to_owned()
}

/// Alias to phik_taylor_ext with k=0
///
/// # Args
/// * `A` : the matrix
pub fn matexp_taylor<T>(a: MatRef<T>) -> Mat<T>
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    phik_taylor_ext(a, 0)
}

/// Optimized phi_k Taylor series for lower-bidiagonal `a_bi`.
///
/// # Args
/// * `a_bi` : the lower bidiagonal matrix
/// * `shift` : spectrum shift parameter. 0.0 for unshifted matexp.
/// * `scale` : spectrum shift parameter. 1.0 for unscaled matexp.
/// * `p` : polynomial order
/// * `k` : phi-fn order
///
pub fn phik_taylor_bidiag<T: ComplexField>(
    a_bi: MatRef<T>,
    shift: f64,
    scale: f64,
    p: usize,
    k: usize,
) -> Mat<T> {
    let n = a_bi.nrows();

    // m = scale * a_bi  — only write the lower-bidiagonal entries, rest stay zero.
    let mut m: Mat<T> = faer::Mat::zeros(n, n);
    {
        let scale_t = from_f64::<T>(scale);
        for i in 0..n {
            let diag_val = a_bi[(i, i)].clone();
            m[(i, i)] = mul(&scale_t, &diag_val);
            if i + 1 < n {
                let sub_val = a_bi[(i + 1, i)].clone();
                m[(i + 1, i)] = mul(&scale_t, &sub_val);
            }
        }
    }

    // ts_expm = I / k!
    let mut ts_expm: Mat<T> = faer::Mat::identity(n, n);
    let mut fact = factorial::factorial(k as u64);
    ts_expm = ts_expm / fact;

    // `bandwidth` = number of active diagonals in `m` (diag + subdiags).
    // Starts at 2 (= diagonal + 1 subdiagonal from scale*a_bi).
    let mut bandwidth: usize = 2_usize.min(n);

    for i in 0..p {
        fact *= (k + i + 1) as f64;
        let inv_fact_t = from_f64::<T>(1.0 / fact);

        // ts_expm += m / fact - band-aware: m[(row,col)] =/= 0 only for col <= row < col+bandwidth.
        for col in 0..n {
            let row_max = (col + bandwidth).min(n);
            for row in col..row_max {
                let elem = mul(&inv_fact_t, &m[(row, col)].clone());
                let old = ts_expm[(row, col)].clone();
                ts_expm[(row, col)] = add(&old, &elem);
            }
        }

        // m <- a_bi * m  in-place via bottom-to-top row sweep.
        // new_m[(r,c)] = d[r]*m[(r,c)] + s[r]*m[(r-1,c)]
        // New bandwidth = bandwidth + 1 (capped at n).
        let new_bw = (bandwidth + 1).min(n);
        for row in (1..n).rev() {
            let d_row = a_bi[(row, row)].clone();
            let s_row = a_bi[(row, row - 1)].clone();
            // Non-zero cols for new m at this row span row.saturating_sub(new_bw-1)..=row.
            let col_start = row.saturating_sub(new_bw - 1);
            for col in col_start..=row {
                let v_rc = m[(row, col)].clone();
                // m[(row-1, col)] is zero when col == row (upper triangle), safe to read.
                let v_prev = m[(row - 1, col)].clone();
                m[(row, col)] = add(&mul(&d_row, &v_rc), &mul(&s_row, &v_prev));
            }
        }
        // Row 0: no subdiagonal contribution.
        {
            let d0 = a_bi[(0, 0)].clone();
            let v00 = m[(0, 0)].clone();
            m[(0, 0)] = mul(&d0, &v00);
        }
        bandwidth = new_bw;
    }

    faer::Scale(from_f64::<T>(shift.exp())) * ts_expm
}

#[derive(Debug)]
pub struct TaylorExpm {
    _order: usize,
}

impl TaylorExpm {
    fn new(&self, p: usize) -> Self {
        Self { _order: p }
    }
}

impl<T> DensePhikvEvaluator<T> for TaylorExpm
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    fn apply_phi_k(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>, k: usize) -> Mat<T> {
        phik_taylor_ext((Scale(from_f64::<T>(dt)) * a).as_ref(), k) * v0
    }
}

#[cfg(test)]
mod test_matexp_taylor {
    use super::*;
    use crate::{
        mat_utils::{mat_mat_approx_eq, random_mat_normal},
        matexp_pade::{matexp, phi_ext},
    };
    use faer::c64;

    #[test]
    fn test_taylor_phi_k() {
        let dense_a: Mat<f64> = random_mat_normal(5, 5);
        for k in 0..=3 {
            let pade_phi_k = phi_ext(dense_a.as_ref(), k);
            let taylor_phi_k = phik_taylor_ext(dense_a.as_ref(), k);
            mat_mat_approx_eq(taylor_phi_k.as_ref(), pade_phi_k.as_ref(), 1e-9);
        }
    }

    #[test]
    fn test_taylor_matexp_real_scaling_and_squaring() {
        let a = faer::mat![[0.0_f64, -4.0], [4.0, 0.0]];

        let taylor = matexp_taylor(a.as_ref());
        let pade = matexp(a.as_ref(), 1.0);

        mat_mat_approx_eq(taylor.as_ref(), pade.as_ref(), 1e-12);
    }

    #[test]
    fn test_taylor_matexp_real_diagonal() {
        let a = faer::mat![[1.0_f64, 0.0], [0.0, -2.0]];

        let taylor = matexp_taylor(a.as_ref());
        let pade = matexp(a.as_ref(), 1.0);

        mat_mat_approx_eq(taylor.as_ref(), pade.as_ref(), 1e-12);
    }

    #[test]
    fn test_taylor_matexp_complex() {
        let a = faer::mat![
            [c64::new(0.0, 0.0), c64::new(0.0, 1.0)],
            [c64::new(0.0, 1.0), c64::new(0.0, 0.0)],
        ];

        let taylor = matexp_taylor(a.as_ref());
        let pade = matexp(a.as_ref(), 1.0);

        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                assert!(
                    (taylor[(i, j)] - pade[(i, j)]).norm() < 1e-12,
                    "entry ({i}, {j}) differs: Taylor = {:?}, Pade = {:?}",
                    taylor[(i, j)],
                    pade[(i, j)]
                );
            }
        }
    }
}
