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
use faer::reborrow::*;
use faer::prelude::*;
use faer::matrix_free::LinOp;
use faer::complex::{ComplexFloat, Complex64};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::traits::ComplexField;
use faer::linalg::matmul::triangular::{matmul as tri_matmul, BlockStructure};
use faer_traits::math_utils::{add, mul, from_f64};

use std::cmp::{max, min};
use statrs::function::{factorial};
use csv;

use crate::ode_sys::{DynRefExtendedLinOp};
use crate::matexp_traits::{LinOpPhikvEvaluator};
use crate::arnoldi::arnoldi_lop_restarted;

/// Pre-generated Leja points from file
/// Real leja points in [-2, 2]
const LEJA_REAL_CSV: &str = std::include_str!("leja_points_real");
/// Complex conjugate leja points are on the unit circle.
const LEJA_CIRCLE_CSV: &str = std::include_str!("leja_points_circle");

/// Generate conjugate-pair leja points along a circle with radius r on the complex plane.
///
/// # Args
/// * `n_leja` : number of leja points to generate
/// * `r` : leja circle radius
///
pub fn gen_leja_circle(n_leja: usize, r: f64) -> (Vec<f64>, Vec<f64>)
{
    let mut leja_re: Vec<f64> = Vec::with_capacity(n_leja);
    let mut leja_im: Vec<f64> = Vec::with_capacity(n_leja);
    let max_base = n_leja.max(2);
    let mut lp_base: Vec<c64> = vec![c64::new(0.0, 0.0); max_base];
    let mut lp_is_real: Vec<bool> = vec![false; max_base];

    // first two leja points {-1+0i, 1+0i}
    lp_base[0] = c64::new(-1.0, 0.0);
    lp_base[1] = c64::new(1.0, 0.0);
    lp_is_real[0] = true;
    lp_is_real[1] = true;

    // root_unity starts at i = exp(i*pi/2).  Each time we finish a full sweep of
    // the upper half-circle we double `full_half_circle` and halve the rotation
    // angle by taking the complex square root of root_unity.
    let mut root_unity = c64::new(0.0, 1.0);
    let mut full_half_circle: usize = 1;

    for lpk in 2..max_base {
        // Rotate the reference point from the previous full half-circle sweep.
        let next_lp = lp_base[lpk - full_half_circle] * root_unity;
        lp_base[lpk] = next_lp;
        lp_is_real[lpk] = false; // LPCONJ: upper half-plane complex point

        // After completing one full sweep, halve the rotation angle.
        if lpk >= 2 * full_half_circle {
            full_half_circle *= 2;
            root_unity = ComplexFloat::sqrt(root_unity);
        }
    }

    // Swap so the sequence opens with +1 then −1
    lp_base.swap(0, 1);
    lp_is_real.swap(0, 1);

    // expand to the full interleaved conjugate-pair sequence
    for i in 0..max_base {
        if leja_re.len() >= n_leja {
            break;
        }
        if lp_is_real[i] {
            // Real anchor point — single entry
            leja_re.push(r * lp_base[i].re());
            leja_im.push(0.0);
        } else {
            // Upper half-plane point
            leja_re.push(r * lp_base[i].re());
            leja_im.push(r * lp_base[i].im());
            // Lower half-plane conjugate (stored immediately after)
            if leja_re.len() < n_leja {
                leja_re.push(r * lp_base[i].re());
                leja_im.push(-r * lp_base[i].im());
            }
        }
    }

    (leja_re, leja_im)
}

/// Generate leja points on the real interval [a, b].
///
/// Ref:
///   Baglama, J., D. Calvetti, and L. Reichel.
///   "Fast leja points." Electron. Trans. Numer. Anal 7.124-140 (1998): 119-120.
///
/// # Args
/// * `n_leja` : number of leja points to generate
/// * `a` : lower  limit, typically -1
/// * `b` : upper limit, typically 1
///
pub fn gen_leja_line(n_leja: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>)
{
    // --- edge cases -------------------------------------------------------
    if n_leja == 0 {
        return (vec![], vec![]);
    }
    let first = if a.abs() > b.abs() { a } else { b };
    if n_leja == 1 {
        return (vec![first], vec![0.0]);
    }
    let second = if a.abs() > b.abs() { b } else { a };
    if n_leja == 2 {
        return (vec![first, second], vec![0.0, 0.0]);
    }

    let n   = n_leja;
    let mid = (a + b) / 2.0;

    // zt[0..3] = first three fast-leja points; remaining entries stay 0.
    let mut zt = vec![0.0_f64; n];
    zt[0] = first;
    zt[1] = second;
    zt[2] = mid;

    // zs[k] = current candidate midpoint whose product-tracker is zprod[k].
    // Entries beyond index 1 stay 0 (inactive) until activated by the loop.
    let mut zs = vec![0.0_f64; n];
    zs[0] = (zt[1] + zt[2]) / 2.0;
    zs[1] = (zt[2] + zt[0]) / 2.0;

    // zprod[k] = prod_j( zs[k] - zt[j] ) over the 3 active seed points.
    let mut zprod = vec![0.0_f64; n];
    zprod[0] = zt[0..3].iter().map(|&z| zs[0] - z).product();
    zprod[1] = zt[0..3].iter().map(|&z| zs[1] - z).product();

    // index[k] = [left_neighbor_idx, right_neighbor_idx] for candidate k.
    let mut index = vec![[0usize; 2]; n];
    index[0] = [1, 2];
    index[1] = [2, 0];

    for i in 3..n {
        // argmax: return the FIRST index that achieves the maximum |zprod|.
        let mut maxi    = 0usize;
        let mut max_val = zprod[0].abs();
        for j in 1..n {
            let v = zprod[j].abs();
            if v > max_val {
                max_val = v;
                maxi    = j;
            }
        }

        // Place new leja point.
        zt[i] = zs[maxi];

        // Update adjacency links.
        index[i-1][0] = i;
        index[i-1][1] = index[maxi][1];
        index[maxi][1] = i;

        // Recompute the two affected candidate midpoints.
        zs[maxi] = (zt[index[maxi][0]] + zt[index[maxi][1]]) / 2.0;
        zs[i-1]  = (zt[index[i-1][0]]  + zt[index[i-1][1]])  / 2.0;

        // Recompute zprod for the two updated candidates using only the
        // i active leja points zt[0..i] (zt[i] was just placed above).
        // This matches Python's  np.prod(zs[k] - zt[0:i])  which uses
        // zt[0:i] (exclusive of the newly placed point at index i).
        zprod[maxi] = zt[0..i].iter().map(|&z| zs[maxi] - z).product();
        zprod[i-1]  = zt[0..i].iter().map(|&z| zs[i-1]  - z).product();

        // Extend all running products by the factor (zs[j] - zt[i]).
        // Matches Python's  zprod = zprod * (zs - zt[i]).
        let zt_i = zt[i];
        for j in 0..n {
            zprod[j] *= zs[j] - zt_i;
        }
    }

    let leja_im = vec![0.0_f64; n];
    (zt, leja_im)
}

/// Rescale the leja points to bound the interval [a, b, -c, +c].
///
/// # Args
/// * `leja_re` : real component of the leja points
/// * `leja_im` : imag component of the leja points
/// * `a` : min real extent
/// * `b` : max real extent
/// * `c` : max imaginary extent
///
/// # Returns:
/// * (leja_re, leja_im, shift, scale)
pub fn shift_scale_leja(leja_re: ColRef<f64>, leja_im: ColRef<f64>, a: f64, b: f64, c: f64)
    -> (Col<f64>, Col<f64>, f64, f64)
{
    assert!(leja_re.nrows() == leja_im.nrows());
    // half axes
    let hax1 = (b - a) / 2.0;
    let hax2 = c;
    let shift = (a + b) / 2.0;
    let scale = (hax1 + hax2) / 2.0;
    // normalize half axes to capacity 1
    let (h1, h2) = (hax1 / scale, hax2 / scale);
    // shift and scale the leja points
    let leja_re_scaled = h1 * leja_re.as_ref();
    let leja_im_scaled = h2 * leja_im.as_ref();
    (leja_re_scaled, leja_im_scaled, shift, scale)
}

/// Inverse operation to shift_scale_leja.  Inverts the shift and scale
/// operation from the bounds [a, b, -c, +c]  back to the original leja
/// sequence bounds.
///
/// # Returns:
/// * (leja_re, leja_im)
pub fn inv_shift_scale_leja(leja_re: ColRef<f64>, leja_im: ColRef<f64>, a: f64, b: f64, c: f64, re_scale: f64, im_scale: f64)
    -> (Col<f64>, Col<f64>)
{
    assert!(b >= a);
    // shift to zero-mean
    let shift = (a + b) / 2.;
    let mut leja_re_s = leja_re - faer::Mat::full(leja_re.nrows(), 1, shift).col(0);
    let mut leja_im_s = leja_im.to_owned();
    // scale
    let width_re = b - a;
    let width_im = 2.0 * c;
    if width_re > f64::EPSILON * 128.0 {
        leja_re_s = 2.0 * re_scale * (&leja_re_s / width_re);
    }
    if width_im > f64::EPSILON * 128.0 {
        leja_im_s = 2.0 * im_scale * (&leja_im_s / width_im);
    }
    (leja_re_s, leja_im_s)
}


/// The Leja points
#[derive(Clone, Debug)]
pub struct LejaPoints {
    leja_re: Col<f64>,
    leja_im: Col<f64>,
    leja_x: Col<c64>,
}

impl LejaPoints {
    pub fn new(leja_re_v: Vec<f64>, leja_im_v: Vec<f64>) -> Self {
        assert!(leja_re_v.len() == leja_im_v.len());
        let n_leja = leja_re_v.len();
        let leja_re: Col<f64> = faer::Col::from_fn(n_leja, |i| {leja_re_v[i]});
        let leja_im: Col<f64> = faer::Col::from_fn(n_leja, |i| {leja_im_v[i]});
        let leja_x: Col<c64> = Col::from_fn(
            n_leja, |i: usize| {c64::new(leja_re_v[i], leja_im_v[i])});

        Self {
            leja_re,
            leja_im,
            leja_x,
        }
    }

    pub fn new_from_col(leja_re: Col<f64>, leja_im: Col<f64>) -> Self {
        assert!(leja_re.nrows() == leja_im.nrows());
        let n_leja = leja_re.nrows();
        let leja_x: Col<c64> = Col::from_fn(
            n_leja, |i: usize| {c64::new(leja_re[i], leja_im[i])});
        Self {
            leja_re,
            leja_im,
            leja_x,
        }
    }

    pub fn new_from_fn(method: &str) -> Self {
        let (lp_re, lp_im) = match method {
            "leja_real" => gen_leja_line(1000, -1.0, 1.0),
            _ => gen_leja_circle(1000, 1.0)
        };
        Self::new(lp_re, lp_im)
    }

    /// Create leja points from csv file of format:
    ///
    /// leja_re_0, leja_im_0
    /// leja_re_1, leja_im_1
    /// ...
    /// leja_re_n, leja_im_n
    ///
    pub fn new_from_file(file_str: &str) -> Self {
        // parse file content string
        todo!("Implement leja points from user file.");
    }

    /// Leja points from pre-generated library
    pub fn new_from_lib(lib_str: &str) -> Self {
        let (lp_str, prescale) = match lib_str {
            "leja_real" => (LEJA_REAL_CSV, 0.5),
            "leja_circle" => (LEJA_CIRCLE_CSV, 1.0),
            _ => panic!("Invalid lib_str.")
            };
        // storage for real and complex leja points
        let mut real_lp: Vec<f64> = vec![];
        let mut complex_lp: Vec<f64> = vec![];
        // parse the leja point string
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(lp_str.as_bytes());
        for result in rdr.records() {
            let record = result.expect("parsing record failed");
            let re: f64 = record.get(0).unwrap().replace(" ", "").parse::<f64>().unwrap();
            let im: f64 = record.get(1).unwrap().replace(" ", "").parse::<f64>().unwrap();
            real_lp.push(re * prescale);
            complex_lp.push(im * prescale);
        }
        Self::new(real_lp, complex_lp)
    }

    /// Number of leja points
    pub fn n_leja(&self) -> usize {
        self.leja_re.nrows()
    }

    /// Number of leading leja points on the real axis
    pub fn n_leja_real(&self) -> usize {
        // count number of leading real leaja points
        let tol = 1.0e-20;
        let mut nr: usize = 0;
        for i in 0..self.leja_re.nrows() {
            if self.leja_im[i].abs() < tol {
                nr += 1;
            }
            else {
                break;
            }
        }
        nr
    }

    /// Number of leading leja points at 0.0 + 0.0j
    pub fn n_leja_zero(&self) -> usize {
        // count number of leading real leaja points
        let tol = 1.0e-25;
        let mut nz: usize = 0;
        for i in 0..self.leja_re.nrows() {
            if self.leja_im[i].abs() < tol && self.leja_re[i].abs() < tol {
                nz += 1;
            }
            else {
                break;
            }
        }
        nz
    }

    /// Build the lower bidiagonal matrix, Xi =
    /// [[leja_0,       0,       0],
    /// [[      1, leja_1,       0],
    /// [[    ...,     ...,    ...],
    /// [[      0,       1, leja_n],
    ///
    pub fn gen_xi(&self) -> Mat<c64> {
        let n_leja = self.leja_re.nrows();
        let mut xi: Mat<c64> = faer::Mat::zeros(n_leja, n_leja);
        for i in 0..n_leja {
            xi[(i, i)] = c64::new(self.leja_re[i], self.leja_im[i]);
            if i+1 < n_leja {
                xi[(i+1, i)] = c64::new(1.0, 0.0);
            }
        }
        xi
    }

    /// Shifted and scaled leja points
    pub fn leja_sc(&self, shift: f64, scale: f64) -> (Col<f64>, Col<f64>) {
        let leja_sc_re = faer::Col::full(self.n_leja(), shift) + scale * self.leja_re.as_ref();
        let leja_sc_im = scale * self.leja_im.as_ref();
        (leja_sc_re, leja_sc_im)
    }

    /// Append additional points to the sequence.
    /// Useful to build a hybrid leja-hermite sequence.
    pub fn append(&self, other: &LejaPoints) -> Self {
        let full_re = faer::concat![[self.leja_re.as_mat()], [other.leja_re.as_mat()]];
        let full_im = faer::concat![[self.leja_im.as_mat()], [other.leja_im.as_mat()]];
        Self::new_from_col(full_re.col(0).to_owned(), full_im.col(0).to_owned())
    }

    /// Concatenate multiple leja sequences together
    pub fn concat(&self, other: Vec<&LejaPoints>) -> Self {
        if other.len() == 0 {
            return self.clone()
        }
        let mut final_lp = self.append(other[0]);
        for i in 1..other.len() {
            final_lp = final_lp.append(other[i])
        }
        final_lp
    }

    /// Prepend `p` zero-spectrum nodes to this Leja sequence.
    ///
    /// Each prepended node has `leja_re = -shift/scale` and `leja_im = 0.0`,
    /// so that `shift + scale * leja_re = 0` in spectrum space.
    /// These nodes encode the BAMPHI Block 1 Taylor prefix: iterating the
    /// Newton recurrence at a zero spectrum node gives `vm <- tau*A*vm/scale`,
    /// i.e., a pure Taylor step.
    ///
    /// Caller must ensure `scale.abs() > 0`.
    ///
    /// # Args
    /// * `p` : number of zeros to prepend
    ///
    pub fn prepend_zeros(&self, p: usize, shift: f64, scale: f64) -> Self {
        assert!(scale.abs() > 0.);
        if p == 0 { return self.clone(); }
        let zero_re = -shift / scale;
        let prefix = LejaPoints::new(vec![zero_re; p], vec![0.0; p]);
        prefix.concat(vec![self])
    }


    /// Mirror Leja points about the real axis
    pub fn mirror(&self) -> Self {
        let mut leja_mirror_re: Vec<f64> = vec![];
        let mut leja_mirror_im: Vec<f64> = vec![];
        for i in 0..self.n_leja() {
            leja_mirror_re.push(self.leja_re[i]);
            leja_mirror_re.push(self.leja_re[i]);
            leja_mirror_im.push(self.leja_im[i]);
            leja_mirror_im.push(-self.leja_im[i]);
        }
        Self::new(leja_mirror_re, leja_mirror_im)
    }

    /// Reorders leja sequence to come in sequence of
    /// real points, then complex conjugate pairs.
    pub fn reorder_conj_pairs(&self) -> Self {
        let mut re_idxs: Vec<usize> = vec![];
        let mut im_idxs: Vec<usize> = vec![];
        let mut leja_reordered: Vec<c64> = vec![];
        let mut leja_re_reordered: Vec<f64> = vec![];
        let mut leja_im_reordered: Vec<f64> = vec![];
        let tol = f64::EPSILON * 100.0;
        // flag real ritz values
        for i in 0..self.leja_re.nrows() {
            if self.leja_im[i].abs() < tol {
                re_idxs.push(i);
            }
            else {
                im_idxs.push(i);
            }
        }
        // reorder leja sequence
        for re_i in re_idxs.iter() {
            leja_reordered.push(c64::new(self.leja_re[*re_i], self.leja_im[*re_i]));
        }
        for im_i in im_idxs.iter() {
            leja_reordered.push(c64::new(self.leja_re[*im_i], self.leja_im[*im_i]));
        }
        for lp in leja_reordered.iter() {
            leja_re_reordered.push((*lp).re());
            leja_im_reordered.push((*lp).im());
        }
        // create reordered leja points
        Self::new(leja_re_reordered, leja_im_reordered)
    }

    /// This method takes the current possibly non-leja sequence of points
    /// on the complex plane and re-orders the points
    /// according to the leja ordering formula.
    /// This allows user to use arbitrary points to generate a leja sequence
    ///
    /// # Args:
    /// * `n` : number of leja points to compute from the available points
    ///
    /// Typically n << self.n_leja()
    pub fn reorder_leja(&self, n: usize) -> Self {
        assert!(n <= self.n_leja());
        todo!();
    }

    /// Normalize the Leja points such that the capacity of the set is 1
    pub fn normalize(&self, a: f64, b: f64, c: f64) -> Self {
        let (leja_re_normed, leja_im_normed) = inv_shift_scale_leja(
            self.leja_re.as_ref(), self.leja_im.as_ref(), a, b, c, 1.0, 1.0);
        Self::new_from_col(leja_re_normed, leja_im_normed)
    }

    /// Extract a slice form the original Leja sequence
    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        let n_end = std::cmp::min(end, self.n_leja());
        Self::new_from_col(self.leja_re.get(start..n_end).to_owned(),
                           self.leja_im.get(start..n_end).to_owned())
    }

    /// Rescale the leja points
    pub fn rescale(&self, a: f64, b: f64, c: f64) -> (Self, f64, f64) {
        let (leja_sc_re, leja_sc_im, shift, scale) = shift_scale_leja(
            self.leja_re.as_ref(), self.leja_im.as_ref(), a, b, c);
        (Self::new_from_col(leja_sc_re, leja_sc_im), shift, scale)
    }
}

/// Compute the dense matrix exponential using tayler series
///
/// # Args
/// * `A` : the matrix
/// * `shift` : spectrum shift parameter. 0.0 for unshifted matexp.
/// * `scale` : spectrum shift parameter. 1.0 for unscaled matexp.
/// * `p` : polynomial order
/// * `k` : phi-fn order
///
pub fn phik_taylor<T: ComplexField>(a: MatRef<T>, shift: f64, scale: f64, p: usize, k: usize) -> Mat<T>
{
    let mut m: Mat<T> = scale * a.as_ref();
    let mut ts_expm: Mat<T> = faer::Mat::identity(m.nrows(), m.ncols());
    let mut fact = factorial::factorial(k as u64);
    ts_expm = ts_expm / fact;
    for i in 0..p {
        fact *= (k + i + 1) as f64;
        ts_expm += m.as_ref() / fact;
        m = a.as_ref() * m.as_ref();
    }
    shift.exp() * ts_expm
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
pub fn phik_taylor_bidiag<T: ComplexField>(a_bi: MatRef<T>, shift: f64, scale: f64, p: usize, k: usize) -> Mat<T>
{
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
                let old  = ts_expm[(row, col)].clone();
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
                let v_rc   = m[(row, col)].clone();
                // m[(row-1, col)] is zero when col == row (upper triangle), safe to read.
                let v_prev = m[(row - 1, col)].clone();
                m[(row, col)] = add(&mul(&d_row, &v_rc), &mul(&s_row, &v_prev));
            }
        }
        // Row 0: no subdiagonal contribution.
        {
            let d0  = a_bi[(0, 0)].clone();
            let v00 = m[(0, 0)].clone();
            m[(0, 0)] = mul(&d0, &v00);
        }
        bandwidth = new_bw;
    }

    faer::Scale(from_f64::<T>(shift.exp())) * ts_expm
}


/// Compute leja divided differences using taylor series method
///
/// Ref: M. Caliari, Padua. Accurate evaluation of divided differences
/// for polynomial interpolation of exponential integrators.
/// Computing. 80. 2007.
///
/// # Args
/// * `leja_x` : the leja points
/// * `shift` : spectrum shift parameter
/// * `scale` : spectrum shift parameter
/// * `h` : substep size
/// * `p` : polynomial order
/// * `k` : phi-fn order
///
pub fn dd_taylor(leja_x: &LejaPoints, shift: f64, scale: f64, h: f64, p: usize, k: usize) -> Col<c64>
{
    let n_leja = leja_x.n_leja();
    let eye = faer::Mat::<c64>::identity(n_leja, n_leja);

    let xi = leja_x.gen_xi();
    let xi_shift: Mat<c64> = shift * eye.as_ref()
        + scale * xi.as_ref();

    // compute mean
    let mu = (h * xi.as_ref()).diagonal().column_vector().sum() / xi.nrows() as f64;

    // shift to zero mean
    let z = xi_shift - faer::Scale(mu) * eye.as_ref();

    // scaling factor (in powers of 2)
    let s_scale = z.norm_max();
    let s = max(( ( s_scale.ln() - (2.0 as f64).ln() ) / (2.0 as f64).ln() ).ceil() as i32, 1);
    let hs = 1.0 / (2.0 as f64).powi(s);

    // compute phi_k(hs*h*Z) — exploits lower-triangular structure of hs*h*z
    let mut f_out = phik_taylor_bidiag((hs*h*z).as_ref(), 0.0, 1.0, p, k);

    // f_out is lower-triangular (result of phik_taylor_bidiag on a lower-triangular input).
    // Use triangular matmul for squaring and matvec to avoid touching the zero upper triangle.
    let alpha = c64::new(1.0, 0.0);

    // squaring
    let total_mvs = (1usize << s) as usize;  // 2^s
    if total_mvs <= n_leja {
        // Cheaper: 2^s triangular matvecs instead of (s-1) matmuls.
        // v is n×1 (Rectangular); f_out is lower-triangular.
        let mut v: Mat<c64> = f_out.col(0).as_mat().to_owned();
        let mut tmp_v: Mat<c64> = faer::Mat::zeros(n_leja, 1);
        for _ in 1..total_mvs {
            tri_matmul(
                tmp_v.as_mut(), BlockStructure::Rectangular,
                faer::Accum::Replace,
                f_out.as_ref(), BlockStructure::TriangularLower,
                v.as_ref(),    BlockStructure::Rectangular,
                alpha, faer::Par::Seq,
            );
            std::mem::swap(&mut v, &mut tmp_v);
        }
        faer::Scale((h * mu).exp()) * v.col(0)
    } else {
        // Standard squaring path with triangular matmul.
        let mut tmp_sq: Mat<c64> = faer::Mat::zeros(n_leja, n_leja);
        for _ in 0..(s - 1) as usize {
            tri_matmul(
                tmp_sq.as_mut(), BlockStructure::TriangularLower,
                faer::Accum::Replace,
                f_out.as_ref(), BlockStructure::TriangularLower,
                f_out.as_ref(), BlockStructure::TriangularLower,
                alpha, faer::Par::Seq,
            );
            std::mem::swap(&mut f_out, &mut tmp_sq);
        }
        // Final step: only need first column of f_out².
        let mut col0: Mat<c64> = f_out.col(0).as_mat().to_owned();
        let mut tmp_v: Mat<c64> = faer::Mat::zeros(n_leja, 1);
        tri_matmul(
            tmp_v.as_mut(), BlockStructure::Rectangular,
            faer::Accum::Replace,
            f_out.as_ref(), BlockStructure::TriangularLower,
            col0.as_ref(),  BlockStructure::Rectangular,
            alpha, faer::Par::Seq,
        );
        faer::Scale((h * mu).exp()) * tmp_v.col(0)
    }
}


/// Compute leja divided differences for phi_k using the dd_phi method
/// of Zivcovich (2019)
///
/// The z interpolation points are kept normalized (O(1) magnitude) and the
/// combined step factor `hs = h * scale` is baked into the Taylor-series seeds
/// as `dd[kk] = hs^kk / (kk! * s^kk)`.  This prevents underflow of the raw
/// divided-difference row when `scale` or `s` is large, without requiring
/// extended precision arithmetic.
///
/// Ref: F. Zivcovich. Fast and accurate computation of divided differences
/// for analytic functions, with an application to the exponential function.
/// Dolomites Research Notes on Approximation. 12. 2019.
///
/// # Args
/// * `leja_x` : the leja points
/// * `shift`  : spectrum shift parameter
/// * `scale`  : spectrum scale parameter
/// * `h`      : substep size
/// * `p`      : taylor series terms (recommend >= 30)
/// * `k`      : phi-fn order
///
pub fn dd_phi(leja_x: &LejaPoints, shift: f64, scale: f64, h: f64, p: usize, k:
usize) -> Col<c64>
{
    let n_leja = leja_x.n_leja();
    let l     = k;              // zeros prepended to handle phi_l
    let total = l + n_leja;     // total interpolation points
    let n     = total - 1;      // highest index, points are 0..=n
    let cap_n = n + p;          // Taylor truncation degree  (N in the paper)

    // Combined step factor used throughout
    let hs = h * scale;

    // z = [0*l, shift/scale + leja_x]  (normalized: O(1) magnitude)
    // The factor hs is NOT baked into z
    let scaled_shift = if scale.abs() > f64::EPSILON { shift / scale } else { 0.0 };
    let mut z: Vec<c64> = vec![c64::new(0.0, 0.0); total];
    for i in 0..n_leja {
        z[l + i] = c64::new(
            scaled_shift + leja_x.leja_re[i],
            leja_x.leja_im[i],
        );
    }

    // Shift normalized z by its mean mu_norm for numerical centering.
    // The true mean (in scaled units) is mu = hs * mu_norm; corrected below.
    let mu_norm: c64 = z.iter().copied()
                        .fold(c64::new(0.0, 0.0), |acc, x| acc + x)
                        * c64::new(1.0 / total as f64, 0.0);
    for zi in z.iter_mut() {
        *zi = *zi - mu_norm;
    }
    // Scale mu back to full units for the final exp(mu) factor
    let mu = mu_norm * c64::new(hs, 0.0);

    // Lower triangle of F contains z[i0] - z[j0] (normalized differences).
    // Track max |entry|, then scale back to full units to compute s.
    let mut f_mat: Mat<c64> = Mat::zeros(total, total);
    let mut max_abs: f64 = 0.0;
    for i0 in 0..n {
        for j0 in (i0 + 1)..=n {
            let val = z[i0] - z[j0];
            f_mat[(j0, i0)] = val;
            let v = val.abs();
            if v > max_abs { max_abs = v; }
        }
    }
    // Correct for the normalization: the true differences are hs * (z[i]-z[j])
    max_abs *= hs;

    // s = max(ceil(max|F_lower_full| / 3.5), 1)
    let s     = (max_abs / 3.5).ceil().max(1.0) as usize;
    let s_f64 = s as f64;

    // Seed dd[kk] = hs^kk / (kk! * s^kk).
    // Baking hs into the seeds compensates for the normalized z-points and
    // prevents underflow when hs is large.
    // Underflow guard: once running_fraction reaches 0 the remaining terms
    // are negligible and stay at the zero-initialised value.
    let mut dd: Vec<c64> = vec![c64::new(0.0, 0.0); cap_n + 1];
    dd[0] = c64::new(1.0, 0.0);
    let mut running_fraction = 1.0_f64;
    for kk in 1..=cap_n {
        running_fraction *= hs / (kk as f64 * s_f64);
        if running_fraction == 0.0 { break; }
        dd[kk] = c64::new(running_fraction, 0.0);
    }

    // H-factorisation sweep — builds F(0) in the upper triangle.
    // z is normalized here; the hs factor lives in dd, so the combined product
    // z[j] * dd[k0+1] is equivalent to z_full[j] * dd_unscaled[k0+1].
    for j in (0..=n).rev() {
        // First inner loop — Taylor remainder sweep
        for k0 in ((n - j)..cap_n).rev() {
            let tmp = z[j] * dd[k0 + 1];
            dd[k0] = dd[k0] + tmp;
        }
        // Second inner loop — divided-difference sweep using F lower triangle.
        // Column j is stored contiguously in faer's column-major layout, so
        // f_mat[(k0+j+1, j)] accesses stride-1 memory as k0 decrements.
        for k0 in (0..(n - j)).rev() {
            let tmp = f_mat[(k0 + j + 1, j)] * dd[k0 + 1];
            dd[k0] = dd[k0] + tmp;
        }
        // Store dd[0..=n-j] into upper-triangle row j of F
        for col in 0..=(n - j) {
            f_mat[(j, j + col)] = dd[col];
        }
        // Zero column j of the lower triangle in-place — entries are no longer
        // needed after this outer iteration, so this replaces the separate triu
        // zeroing pass that previously followed the sweep.
        for row in (j + 1)..=n {
            f_mat[(row, j)] = c64::new(0.0, 0.0);
        }
    }

    // Overwrite diagonal: F[i,i] = exp(hs * z_norm[i] / s)
    // hs applied explicitly here because z is normalized (not fully scaled).
    let hs_over_s = c64::new(hs / s_f64, 0.0);
    for i in 0..=n {
        f_mat[(i, i)] = (hs_over_s * z[i]).exp();
    }

    // Squaring phase — pre-allocate two (1×total) row buffers and ping-pong
    // using faer::linalg::matmul::matmul (Accum::Replace).
    let mut buf_a: Mat<c64> = Mat::from_fn(1, total, |_, j| f_mat[(0, j)]);
    if s > 1 {
        let mut buf_b: Mat<c64> = Mat::zeros(1, total);
        for _ in 0..(s - 1) {
            faer::linalg::matmul::matmul(
                buf_b.as_mut(),
                faer::Accum::Replace,
                buf_a.as_ref(),
                f_mat.as_ref(),
                c64::new(1.0, 0.0),
                faer::Par::Seq,
            );
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
    }
    let dd_row = buf_a;

    // Output: exp(mu) * dd_row[l+i].
    let exp_mu = mu.exp();
    Col::from_fn(n_leja, |i| {
        exp_mu * dd_row[(0, l + i)]
    })
}


/// Used for phi function evaluation at the leja points
/// Evaluates linear combinations of phi-function-vector products
/// using either the real leja point method (RelPM) or
/// the conj. complex conj leja point method (CLaPM).
pub struct LejaPhiEval {
    /// the leja points
    leja_x: LejaPoints,
    /// the base leja points
    leja_base: LejaPoints,
    /// maximum leja polynomial degree
    m: usize,
    /// number of prepended zeros to use in the leja sequence
    p: usize,
    /// convergence criteria of the Leja polynomial
    tol: f64,
    abort_tol: f64,
    /// Max number of substeps to take when computing the matrix exponential
    max_substeps: usize,
    /// Flag to re-use any krylov subspace information in the
    /// Leja polynomial interpolation procedure.
    krylov_reuse: bool,
    /// Polynomial approximation method. One of: clapm, relpm, taylor
    method: String,
    /// Divided difference method
    dd_method: String,
    /// Extracts spectrum information from a LinOp
    leja_ellipse_adapter: Box<dyn GetSpectrumBounds>,
}


/// Leja point phi function evaluator
impl LejaPhiEval {
    /// Create a new leja point phi function evaluator
    ///
    /// #Args
    /// * `leja_x` - the leja points
    /// * `m` - maximum polynomial approximation order
    /// * `tol` - leja polynomial approximation tolerance
    /// * `dd_method` - method used to compute divided differences
    /// * `krylov_reuse` - reuse krylov subspace for fast interpolation at the ritz values
    /// * `leja_ellipse_adapter` - Spectrum adapter to auto scale leja ellipse
    ///
    pub fn new(
        leja_x: LejaPoints,
        m: usize,
        tol: f64,
        method: &str,
        dd_method: &str,
        krylov_reuse: bool,
        leja_ellipse_adapter: Box<dyn GetSpectrumBounds>
        ) -> Self
    {
        Self {
            m: m,
            p: 0,
            leja_x: leja_x.clone(),
            leja_base: leja_x,
            tol: tol,
            abort_tol: 1e26,
            max_substeps: 0,
            krylov_reuse: krylov_reuse,
            method: method.to_string(),
            dd_method: dd_method.to_string(),
            leja_ellipse_adapter: leja_ellipse_adapter
        }
    }

    /// The Real leja point method (ReLPM) method of
    ///
    /// Caliari, Marco, Marco Vianello, and Luca Bergamaschi.
    /// "Interpolating discrete advection–diffusion propagators at
    /// Leja sequences."
    /// Journal of Computational and Applied Mathematics 172.1 (2004): 79-99.
    ///
    /// L. Bergamaschi.  M. Caliari. A. Martinez and M. Vianello.
    /// Leja and Krylov Approximations of Large Scale
    /// Matrix Exponentials. Intl. Conf on Computational Science. 2006.
    ///
    /// Computes the matrix exponential-vector product: exp(tau*dt*A)*u
    ///
    /// #Args
    /// * `pm` - the output vector holding the polynomial approximation of
    ///    the matrix exponential-vector product.
    /// * `ext_a_lo` - the linear operator A
    /// * `tau` - the substep size in [0, 1]
    /// * `u` - the rhs vector
    /// * `shift` - the leja point sequence shift
    /// * `scale` - the leja point sequence scale
    /// * `coeffs` - the leja polynomial coefficients
    fn real_leja_expmv<T: LinOp<f64>>(
        &self,
        mut pm: faer::MatMut<f64>,
        ext_a_lo: &T,
        tau: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        coeffs: ColRef<c64>,
        use_krylov: bool,
        ) -> (bool, usize)
    {
        log::info!("=== ReLPM, shift: {:0.6e}, scale: {:0.6e}", shift, scale);
        let clock = std::time::Instant::now();
        let mut iter: usize = 0;
        let norm_u: f64 = u.norm_l2();
        let mut err_est: f64 = 2. * norm_u;
        let mut converged: bool = err_est == 0.;

        // shift and scale leja points to align to the spectrum parameters
        let (leja_x_sc, _leja_x_sc_im) = self.leja_x.leja_sc(shift, scale);

        // first term of leja polynomial
        pm.copy_from(coeffs[0].re * u);
        let mut vm = u.to_owned();
        let mut av = u.to_owned();

        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(ext_a_lo.apply_scratch(u.ncols(), par));

        // Augment leja sequence with krylov subspace polynomial if available
        let krylov_res = if use_krylov {
            self.krylov_poly_expmv(None, tau,
                leja_x_sc.as_ref(), _leja_x_sc_im.as_ref(),
                coeffs, shift, scale, norm_u)
        } else { Err(()) };
        let mut r: usize = 0;  // number of ritz values
        match krylov_res {
            Ok((n_r, xr, dr)) => {
                pm.copy_from(xr);
                vm = dr;
                r = n_r;
            }
            _ => {
                if self.p > 0 {
                    pm += faer::Scale(coeffs[self.p].re) * vm.as_ref();
                }
            }
        }

        // compute leja poly and check for convergence each iter
        for i in 1+r..self.m {
            if converged {
                break;
            }
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                par,
                MemStack::new(&mut mem_buf)
                );
            vm = (tau * av.as_ref() - leja_x_sc[i-1]*vm) / scale;
            // leja polynomial update
            pm += coeffs[i].re * vm.as_ref();

            // check error estimate
            err_est = (coeffs[i].re * vm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
            log::info!("real, {i}, {:0.8e} + {:0.8e}i, {:0.6e}, {:0.6e}",
                leja_x_sc[i-1], 0.0, coeffs[i], err_est);
            if err_est > self.abort_tol {
                println!("Hit abort tol: {err_est:0.2e}. Consider a smaller step.");
                break;
            }
        }
        println!("ReLPM time (s): {}", clock.elapsed().as_secs_f64());
        (converged, iter)
    }

    /// Taylor series method to estimate the action of
    /// the matrix exponential on a vector.
    ///
    /// Computes the matrix exponential-vector product: exp(tau*A)*u
    ///
    /// #Args
    /// * `pm` - the output vector holding the polynomial approximation of
    ///    the matrix exponential-vector product.
    /// * `ext_a_lo` - the linear operator A
    /// * `tau` - the substep size in [0, 1]
    /// * `u` - the rhs vector
    /// * `shift` - location on the real axis about which the
    ///    taylor expansion is conducted.
    /// * `scale` - unused
    fn taylor_expmv(
        &self,
        mut pm: faer::MatMut<f64>,
        ext_a_lo: &dyn LinOp<f64>,
        tau: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        m: usize
        ) -> (bool, usize, Mat<f64>)
    {
        log::info!("=== TS, shift: {:0.6e}, scale: {:0.6e}", shift, scale);
        let clock = std::time::Instant::now();
        let mut iter: usize = 0;
        let norm_u: f64 = u.norm_l2();
        let mut err_est = 2.0 * norm_u;
        let mut converged: bool = err_est == 0.0;

        let mut av = u.to_owned();
        let mut vm = u.to_owned();

        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(ext_a_lo.apply_scratch(u.ncols(), par));

        // compute first term of the taylor polynomial
        // ext_a_lo.apply(pm, u, par, mem_scratch);
        let mut coeff = 1.0;
        pm.copy_from(coeff * u);

        for j in 1..m {
            if converged {
                break;
            }
            coeff = 1.0 / factorial::factorial(j as u64);
            let mem_scratch = MemStack::new(&mut mem_buf);
            ext_a_lo.apply(av.as_mut(), vm.as_ref(), par, mem_scratch);
            vm = tau * av.as_ref();
            pm += coeff * vm.as_ref();

            // check error estimate
            err_est = (coeff * vm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
            log::info!("tayl, {j}, {:0.8e} + {:0.8e}i, {:0.6e}, {:0.6e}", 0.0, 0.0, coeff, err_est);
        }
        println!("TS time (s): {}", clock.elapsed().as_secs_f64());
        (converged, iter, vm)
    }

    /// Compute the augmenting first term in the krylov-leja sequence
    /// with krylov subspace polynomial, if available.
    ///
    /// If the krylov subspace has not been computed, or is unavailable,
    /// this routine returns Err(()).
    ///
    /// Ref:
    /// Caliari, Marco, Fabio Cassini, and Franco Zivcovich.
    /// "BAMPHI: Matrix-free and transpose-free action of linear combinations
    /// of phi-functions from exponential integrators."
    /// Journal of Computational and Applied Mathematics 423 (2023): 114973.
    ///
    /// #Args:
    /// * `dm` - the initial iteration vector.  None if no initial vector
    /// * `tau` - the substep size in (0, 1]
    /// * `tau_rho_re` - the scaled ritz values real component
    /// * `tau_rho_im` - the scaled ritz values imag component
    /// * `coeffs` - the divided differences.  Newton polynomial coeffs
    ///
    fn krylov_poly_expmv(
        &self,
        dm: Option<Mat<f64>>,
        tau: f64,
        tau_rho_re: ColRef<f64>,
        tau_rho_im: ColRef<f64>,
        coeffs: ColRef<c64>,
        _shift: f64,
        scale: f64,
        norm_u: f64,
        )
        -> Result<(usize, Mat<f64>, Mat<f64>), ()>
    {
        match (self.krylov_reuse, self.leja_ellipse_adapter.get_krylov_subspace()) {
            (true, (Some(q), Some(h))) => {
                // number of ritz values available
                let n_r = h.nrows()-1;

                // convert to complex for interpolation at the ritz values
                // Note: The hessenberg matrix h built from dt*A within
                // the extend linop \tilde A = [[dt*A, B], [0, K]]
                let cmplx_h: Mat<c64> = faer::Mat::from_fn(
                    h.nrows(), h.ncols(), |i, j| { tau*c64::new(h[(i, j)], 0.0) } );

                let gamma = c64::new(scale, 0.0);
                let mut dr = match dm {
                    Some(dm) => {
                        // convert initial iteration vector to complex
                        let d0 = faer::Mat::from_fn(
                            dm.nrows(), dm.ncols(), |i, j| { c64::new(dm[(i, j)], 0.0) } );
                        d0
                    },
                    _ => {
                        let mut e1: Mat<c64> = Mat::zeros(h.nrows(), 1);
                        e1[(0, 0)] = c64::new(1.0, 0.0);
                        e1
                    }
                };

                // compute the first n_r polynomial terms
                let mut pm = faer::Scale(coeffs[self.p]) * dr.as_ref();
                for r in self.p+1..=self.p+n_r {
                    log::info!("kryl, {r}, {:0.8e} + {:0.8e}i, {:0.6e}, {:0.6e}",
                                tau_rho_re[r-1], tau_rho_im[r-1], coeffs[r], 0.);
                    let z = c64::new(tau_rho_re[r-1], tau_rho_im[r-1]);
                    dr = (cmplx_h.as_ref()*dr.as_ref() - faer::Scale(z)*dr.as_ref()) / faer::Scale(gamma);
                    pm += faer::Scale(coeffs[r]) * dr.as_ref();
                }

                // convert to reals and project back to full space.
                // Scale by norm_u so that pr_re = norm_u * Q * Re(pm) approximates exp(tau*A)*u.
                let pm_re = faer::Mat::from_fn(
                    pm.nrows(), pm.ncols(), |i, j| { pm[(i, j)].re } );
                let dr_re = faer::Mat::from_fn(
                    dr.nrows(), dr.ncols(), |i, j| { dr[(i, j)].re } );

                // q is (n, bdwn+1); pm_re/dr_re are (bdwn+1, 1) — shapes are compatible.
                // The projected result lives in pure-A space (n rows).  Pad with self.p zeros
                // so the returned vectors match the extended linop dimension (n + p) expected
                // by pm and vm in complex_conj_leja_expmv.  Zero-padding is exact because
                // K^p = 0 means the nilpotent block contributes nothing.
                let n_q = q.nrows();
                let pr_proj = norm_u * q.as_ref() * pm_re.as_ref();
                let dr_proj = norm_u * q.as_ref() * dr_re.as_ref();

                let mut pr_full = faer::Mat::zeros(n_q + self.p, 1);
                pr_full.get_mut(0..n_q, ..).copy_from(pr_proj.as_ref());
                let mut dr_full = faer::Mat::zeros(n_q + self.p, 1);
                dr_full.get_mut(0..n_q, ..).copy_from(dr_proj.as_ref());

                Ok((n_r, pr_full, dr_full))
            },
            _ => Err(())
        }
    }

    /// Complex conjugate leja point method (CLaPM).
    ///
    /// Computes the matrix exponential-vector product: exp(tau*A)*u
    ///
    /// #Args
    /// * `pm` - the output vector holding the polynomial approximation of
    ///    the matrix exponential-vector product.
    /// * `ext_a_lo` - the linear operator A prescaled by dt
    /// * `tau` - the substep size in [0, 1]
    /// * `u` - the rhs vector
    /// * `shift` - the leja point sequence shift
    /// * `scale` - the leja point sequence scale
    /// * `coeffs` - the leja polynomial coefficients
    fn complex_conj_leja_expmv<T: LinOp<f64>>(
        &self,
        mut pm: faer::MatMut<f64>,
        ext_a_lo: &T,
        tau: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        coeffs: ColRef<c64>,
        use_krylov: bool,
        ) -> (bool, usize)
    {
        let clock = std::time::Instant::now();
        log::info!("=== CLaPM, shift: {:0.6e}, scale: {:0.6e}", shift, scale);
        let mut iter: usize = 0;

        // short-circuit to taylor if scale and shift small
        if (shift.abs()<f64::EPSILON && scale.abs()<f64::EPSILON) || self.method.as_str() == "taylor" {
            let (conv, iter, _) =  self.taylor_expmv(pm, ext_a_lo, tau, u, shift, scale, self.m);
            return (conv, iter)
        }
        // use the real leja point method if all leja points are on the real line
        if self.leja_x.n_leja_real() >= self.m {
            return self.real_leja_expmv(
                pm, ext_a_lo, tau, u, shift, scale, coeffs, use_krylov)
        }

        // shift and scale leja points to align to the spectrum parameters
        let (leja_x_sc_re, leja_x_sc_im) = self.leja_x.leja_sc(shift, scale);

        // build taylor polynomial for p number of iterations
        // let mut tvm: Option<Mat<f64>> = None;
        let mut vm = u.to_owned();
        if self.p > 0 {
            let (_, _, tay_vm) = self.taylor_expmv(
                pm.rb_mut(), ext_a_lo, tau, u, shift, scale, self.p);
            vm = tay_vm;
        }
        else {
            // first term of leja polynomial
            pm.copy_from( coeffs[0].re * u );
        }
        let mut av = u.to_owned();
        let mut qm = u.to_owned();
        let mut nv = u.to_owned();

        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(ext_a_lo.apply_scratch(u.ncols(), par));

        // Fix A: advance vm one more scaled step so that vm = vₚ = (tau/scale)^p * Ã^p * u.
        // taylor_expmv(m=p) loops for j in 1..p, returning tau^{p-1} * Ã^{p-1} * u.
        // One more application gives tau^{p-1} * Ã^p * u; scaling by tau/scale^p yields vₚ.
        if self.p > 0 {
            ext_a_lo.apply(av.as_mut(), vm.as_ref(), par, MemStack::new(&mut mem_buf));
            let scale_p = tau / scale.powi(self.p as i32);
            vm = faer::Scale(scale_p) * av.as_ref();
        }

        let norm_u: f64 = vm.norm_l2();
        let mut err_est = 2. * norm_u;
        let mut converged: bool = err_est == 0.;

        // Augment leja sequence with krylov subspace polynomial if available
        let krylov_res = if use_krylov {
            self.krylov_poly_expmv(None, tau,
                leja_x_sc_re.as_ref(), leja_x_sc_im.as_ref(),
                coeffs, shift, scale, norm_u)
        } else { Err(()) };
        let mut rp: usize = 0;
        match krylov_res {
            Ok((nr, pr, dr)) => {
                if self.p > 0 {
                    pm.copy_from(pr + pm.as_ref());
                } else {
                    pm.copy_from(pr);
                }
                // iteration vector
                vm.copy_from(dr);
                // number of ritz values used
                rp = nr;
            }
            _ => {
                if self.p > 0 {
                    pm += faer::Scale(coeffs[self.p].re) * vm.as_ref();
                }
            }
        }
        // number of leading zeros + n ritz values
        rp += self.p;
        // extract next leja points in the sequence
        let n_leja_real = self.leja_x.slice(rp, rp+10).n_leja_real();

        // precompute scaling factors
        let inv_scale    = 1.0 / scale;
        let tau_inv_scale = tau * inv_scale;

        // compute leja polynomial terms for leading real points
        for i in 1+rp..=n_leja_real+rp {
            if converged {
                break;
            }
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                par,
                MemStack::new(&mut mem_buf)
                );
            // leja polynomial update
            // vm = (tau * av.as_ref() - leja_x_sc_re[i-1]*vm) / scale;
            let z_re_inv = leja_x_sc_re[i-1] * inv_scale;
            faer::zip!(&mut vm, &av).for_each(|faer::unzip!(mut v, a)| {
                *v = tau_inv_scale * *a - z_re_inv * *v;
            });
            let c_re = coeffs[i].re;
            pm += faer::Scale(c_re) * vm.as_ref();

            // check error estimate
            err_est = (c_re * vm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
            log::info!("real, {i}, {:0.8e} + {:0.8e}i, {:0.6e}, {:0.6e}",
                leja_x_sc_re[i-1], leja_x_sc_im[i-1], coeffs[i], err_est);
        }

        // compute remaining leja polynomial terms suported at
        // conjugate complex points.
        for i in (n_leja_real+1+rp..self.m-1).step_by(2) {
            if converged {
                break;
            }
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                par, MemStack::new(&mut mem_buf));

            let z_re_inv = leja_x_sc_re[i-1] * inv_scale;
            let im_sq    = (leja_x_sc_im[i-1] * inv_scale).powi(2);
            let c_re     = coeffs[i].re;
            let c1_re    = coeffs[i+1].re;

            // qm = (tau * av - z_re * vm) * inv_scale
            // in-place, no alloc
            faer::zip!(&mut qm, &av, &vm).for_each(|faer::unzip!(mut q, a, v)| {
                *q = tau_inv_scale * *a - z_re_inv * *v;
            });
            pm += faer::Scale(c_re) * qm.as_ref();

            ext_a_lo.apply(
                av.as_mut(), qm.as_ref(),
                par, MemStack::new(&mut mem_buf));

            // nv = (tau * av - z_re * qm) * inv_scale + im_sq * vm
            // in-place, no alloc
            faer::zip!(&mut nv, &av, &qm, &vm).for_each(
                |faer::unzip!(mut n, a, q, v)|
                {
                    *n = tau_inv_scale * *a - z_re_inv * *q + im_sq * *v;
                });
            std::mem::swap(&mut vm, &mut nv);

            pm += faer::Scale(c1_re) * vm.as_ref();

            err_est = (vm.norm_l2() * c1_re).abs();
            converged = err_est < self.tol * norm_u;
            iter += 2;

            log::info!("cclp, {}, {:0.8e} + {:0.8e}i, {:0.6e}, {:0.6e}",
                i, leja_x_sc_re[i-1], leja_x_sc_im[i-1], coeffs[i], err_est);
            log::info!("cclp, {}, {:0.8e} + {:0.8e}i, {:0.6e}, {:0.6e}",
                i+1, leja_x_sc_re[i], leja_x_sc_im[i], coeffs[i+1], err_est);
            if err_est > self.abort_tol {
                println!("Hit abort tol: {err_est:0.2e}. Consider a smaller step.");
                break;
            }
        }

        println!("CLaPM time (s): {}", clock.elapsed().as_secs_f64());
        (converged, iter)
    }

    /// compute leja poly coeffs by divided difference
    fn leja_poly_coeffs(&self, lp: &LejaPoints, shift: f64, scale: f64, h: f64) -> Col<c64>
    {
        if self.method == "taylor" {
            return faer::Col::zeros(1)
        }
        let clock = std::time::Instant::now();
        let coeffs: Col<c64> = if self.dd_method == "dd_phi" {
            print!("Running dd_phi. ");
            log::info!("Running dd_phi divided difference calc.");
            dd_phi(lp, shift, scale, h, 32, 0)
        }
        else {
            print!("Running dd_taylor. ");
            log::info!("Running dd_taylor divided difference calc.");
            dd_taylor(lp, shift, scale, h, 16, 0)
        };
        log::info!("divided difference walltime (s): {}", clock.elapsed().as_secs_f64());
        println!("divided difference walltime (s): {}", clock.elapsed().as_secs_f64());
        coeffs
    }

    /// Computes the linear combination: phi_0(h*A)*v_0 + ... phi_k(h*A)*v_k
    /// by leja polynomial approximation with optional substepping
    ///
    /// #Args
    /// * `ext_a_lo` - the linear operator A
    /// * `h` - the stepsize, typically h=1.0 if linop A has dt pre-multiplied into it
    /// * `vb` - a k-len sequence of rhs vectors corrosponding to each phi-function: phi_k
    pub fn leja_expmv_substep(&self, ext_a_lo: &DynRefExtendedLinOp, h: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64>
    {
        // remark: ext_a_lo may contain a scaling by dt, so ext_a_lo = dt*A
        // setup the extended rhs vector
        let (mut w_t, n) = ext_a_lo.get_v(vb);

        // allocate storage for result
        let mut w: Mat<f64> = faer::Mat::zeros(w_t.nrows(), w_t.ncols());

        // the leja points
        let lp = self.leja_x.slice(0, self.m);

        // shift and scale parameters
        let (shift, scale) = self.leja_ellipse_adapter.get_shift_scale();

        if self.max_substeps == 0 {
            // compute newton polynomial coefficients of exp(z_sc) with z_sc = shift + scale*z
            let coeffs = self.leja_poly_coeffs(&lp, shift, scale, 1.0);

            // no substep
            let (_conv, _iters) = self.complex_conj_leja_expmv(
                w.as_mut(), ext_a_lo, h, w_t.as_ref(), shift, scale,
                coeffs.as_ref(), true);
            println!("converged: {}, leja iters: {}, shift: {}, scale: {}",
                _conv, _iters, shift, scale);
        } else {
            // Substep the solution y_n+1 = exp(tau * h * A)*y_n
            // where tau is the substep size
            let tau = 1.0 / self.max_substeps as f64;
            let h_tau = h * tau;
            let shift_tau = shift * tau;
            let scale_tau = scale * tau;
            // use the leja points scaled down by tau to evaluate the divided differences
            // for the smaller spectrum of tau*h*A  compared to the full step h*A
            let coeffs = self.leja_poly_coeffs(&lp, shift_tau, scale_tau, 1.0);
            for i in 0..self.max_substeps {
                println!("substep: {} / {}", i+1, self.max_substeps);

                let (_conv, _iters) = self.complex_conj_leja_expmv(
                    w.as_mut(), ext_a_lo, h_tau, w_t.as_ref(),
                    shift_tau, scale_tau, coeffs.as_ref(), i == 0);

                println!("sub converged: {}, leja iters: {}, shift: {}, scale: {}",
                    _conv, _iters, shift_tau, scale_tau);

                // update current solution substep vector
                if i < self.max_substeps - 1 {
                    w_t = w.cloned();
                }
            }
        }

        // extract the first n elements
        w.get(0..n, 0..1).to_owned()
    }

    /// Log optional performance and accuracy metrics of the polynomial interpolation.
    /// NOTE: Very expensive to run. Only intended for debugging or diagnostic mode.
    fn leja_performance_detail(&self, ext_a_lo: &DynRefExtendedLinOp, h: f64, vb: &Vec<MatRef<f64>>)
    {
        // compute the approximation accuracy of leja polynomial interpolation
        // || exp(h*Lambda)*v - p_m_leja(Lambda, h, v) ||
        // with increasing approximation order m
        // where Lambda is a digonal matrix of eigs(J) where J is
        // the system jacobian.  Lambda is estimated via Krylov Shur since
        // J is provided as a LinOp.
        //
        // compute Lambda
        let (ext_v, n) = ext_a_lo.get_v(vb);
        let (_, _, _, lambda_re, lambda_im, _ev) =
            spectrum_krylov_schur(ext_a_lo, ext_v.as_ref(), 1.0, 100, 1e-8, false);
        // compute exp(h*lambda_i)*v_i
        let expected: Vec<c64> = lambda_re.iter().zip(lambda_im.iter()).map(
            |(x_re, x_im)| c64::new(*x_re, *x_im).exp()).collect();
        // compute p_m_leja(h*Lambda)*v
        // create diagonal matrix Lambda
        let lambda = faer::Mat::from_fn(lambda_re.len(), lambda_re.len(),
            |i, j| {
                if i == j {
                   c64::new(lambda_re[i], lambda_im[i])
                }
                else {
                   c64::new(0.0, 0.0)
                }
            }
            );

    }

    /// Set the krylov reuse flag
    pub fn set_krylov_reuse(&mut self, krylov_reuse: bool) {
        self.krylov_reuse = krylov_reuse;
    }

    /// Set the max leja polynomial degree
    pub fn set_m(&mut self, m: usize) {
        self.m = m;
    }

    /// Set the max number of substeps
    pub fn set_max_substeps(&mut self, max_substeps: usize) {
        self.max_substeps = max_substeps;
    }

    /// Set the shift and scale parameters and adjust leja sequence
    /// by splicing in additional points.
    ///
    /// #Args
    /// * `a` - is min real spectrum eig
    /// * `b` - is max real spectrum eig
    /// * `c` - is max imag spectrum eig magnitude
    /// * `p` - number of zeros to prepend to the sequence
    /// * `splice_idx` - index where splice_lp are inserted
    /// * `splice_lp` - optional sequence of points to splice into the full sequence
    pub fn update_leja_splice(
        &mut self, a: f64, b: f64, c: f64, p: usize, splice_idx: usize, splice_lp: LejaPoints)
    {
        let (leja_x, shift, scale) = self.leja_base.rescale(a, b, c);
        // construct the full leja sequence by splicing
        let first_lp = leja_x.slice(0, splice_idx);
        let last_lp = leja_x.slice(splice_idx, leja_x.n_leja());
        // splice into final sequence
        let mut leja_x_ext = first_lp.concat(vec![&splice_lp, &last_lp]);
        if p > 0 {
            leja_x_ext = leja_x_ext.prepend_zeros(p, shift, scale);
            self.p = p;
        }
        self.leja_x = leja_x_ext;
    }

    /// Set the shift and scale parameters
    ///
    /// #Args
    /// * `a` - is min real spectrum eig
    /// * `b` - is max real spectrum eig
    /// * `c` - is max imag spectrum eig magnitude
    /// * `p` - number of zeros to prepend to the sequence
    pub fn update_leja(&mut self, a: f64, b: f64, c: f64, p: usize) {
        let (mut leja_x, shift, scale) = self.leja_base.rescale(a, b, c);
        if p > 0 {
            leja_x = leja_x.prepend_zeros(p, shift, scale);
            self.p = p;
        }
        self.leja_x = leja_x;
    }

}

impl LinOpPhikvEvaluator for LejaPhiEval {
    fn apply_phi_k_v(&mut self, a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64> {
        let clock = std::time::Instant::now();
        // TODO: optionally auto-run apply_prepare here!
        // remark: a_lo may contain a scaling by dt, so a_lo = dt*A
        let res = self.leja_expmv_substep(a_lo, dt, vb);
        println!("apply time (s): {}", clock.elapsed().as_secs_f64());
        res
    }

    fn apply_phi_k(&self, a_lo: &dyn LinOp<f64>, dt: f64, v: MatRef<f64>, k: usize) -> Mat<f64> {
        // create an extended linop
        let mut vbk: Vec<MatRef<f64>> = vec![];
        let tmp_zeros = faer::Mat::zeros(v.nrows(), v.ncols());
        for _i in 0..k {
            vbk.push(tmp_zeros.as_ref());
        }
        vbk.push(v);
        // remark: ext_a_lo contains a scaling by dt, so ext_a_lo = dt*A
        let ext_a_lo = DynRefExtendedLinOp::new(dt, a_lo, &vbk);
        // TODO: optionally auto-run apply_prepare here!
        // compute phi_k(a_lo)*v
        self.leja_expmv_substep(&ext_a_lo, dt, &vbk)
    }

    fn apply_prepare(&mut self, a_lo: &dyn LinOp<f64>, dt: f64, v: MatRef<f64>, k: usize) {
        let clock = std::time::Instant::now();
        println!("=== Updating Spectrum Parameters ===");
        self.leja_ellipse_adapter.update(a_lo, v.as_ref(), dt);
        let (a, b, c) = self.leja_ellipse_adapter.get_bounds();
        let p = k;
        match (self.krylov_reuse, self.leja_ellipse_adapter.get_ritz_leja()) {
            (true, Some(lp_ritz)) => {
                // inject the ritz values into the leja sequence
                self.update_leja_splice(a, b, c, p, 0, lp_ritz)
            },
            _ => { self.update_leja(a, b, c, 0); }
        }
        println!("apply_prepare time (s): {}", clock.elapsed().as_secs_f64());
    }
}


/// Methods for spectrum adapters
pub trait GetSpectrumBounds {
    /// updates the spectrum bounds
    fn update(&mut self, ext_a_lo: &dyn LinOp<f64>, v0: MatRef<f64>, scale: f64);

    /// get the spectrum bounds
    fn get_bounds(&self) -> (f64, f64, f64);

    /// get the leja shift and scale parameters
    fn get_shift_scale(&self) -> (f64, f64);

    /// get the ritz values.  Default behavior assumes ritz values are not available
    fn get_ritz(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) { (None, None) }

    /// Number of available ritz values
    fn n_ritz(&self) -> usize { 0 }

    /// get the ritz values in a leja reordering.
    /// Default behavior assumes ritz values are not available
    fn get_ritz_leja(&self) -> Option<LejaPoints> {
        None
    }

    /// get the hessenberg matrix.  Default behavior assumes krylov subspace is not available
    fn get_krylov_subspace(&self) -> (Option<Mat<f64>>, Option<Mat<f64>>) {
        (None, None)
    }
}


/// Extracts spectrum information from a LinOp using Arnoldi iteration
#[derive(Clone, Debug)]
pub struct LejaEllipseAdapterArnoldiIOM {
    // spectrum parameters
    a: f64,
    b: f64,
    c: f64,
    // spectrum analysis settings
    spec_norm: f64,
    spec_norm_tol: f64,
    spec_iters: usize,
    spec_iom: usize,
    spec_saftey_factor: f64,
    // storage for Krylov subspace and ritz values
    arnld_q: Option<Mat<f64>>,
    arnld_h: Option<Mat<f64>>,
    ritz_re: Option<Vec<f64>>,
    ritz_im: Option<Vec<f64>>,
}

impl LejaEllipseAdapterArnoldiIOM {
    /// Initialize spectrum analysis
    ///
    /// #Args
    /// * `spec_norm_tol` - tolerance used to trigger recomputation of spectrum parameters
    /// * `spec_iters` - maximum number of arnoldi iterations used in spectrum parameter calc
    /// * `spec_iom` - orthogonalization depth used in arnolid spectrum parameter estimate
    /// * `spec_method` - method used to estimate spectrum parameters
    pub fn new(
        a: f64, b: f64, c: f64,
        spec_norm_tol: f64, spec_iters: usize, spec_iom: usize, spec_saftey_factor: f64)
    -> Self
    {
        assert!(a <= b);
        assert!(c >= 0.0);
        Self {
            a,
            b,
            c,
            spec_norm: 0.0,
            spec_norm_tol,
            spec_iters,
            spec_iom,
            spec_saftey_factor: spec_saftey_factor,
            arnld_q: None,
            arnld_h: None,
            ritz_re: None,
            ritz_im: None,
        }
    }
}

impl GetSpectrumBounds for LejaEllipseAdapterArnoldiIOM {
    fn update(&mut self, a_lo: &dyn LinOp<f64>, v: MatRef<f64>, scale: f64)
    {
        let ones = faer::Mat::ones(a_lo.nrows(), 1);
        let mut av = faer::Mat::zeros(a_lo.nrows(), 1);
        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(a_lo.apply_scratch(v.ncols(), par));
        a_lo.apply(
            av.as_mut(),
            ones.as_ref(),
            par,
            MemStack::new(&mut mem_buf));
        let spec_norm = av.norm_l2();

        // only recompute a_lo spectrum parameters if norm has changed
        if (spec_norm - self.spec_norm).abs() > self.spec_norm_tol
        {
            let (fit_a, fit_b, fit_c, ritz_re, ritz_im, q, h) = spectrum_arnoldi_iom(
                a_lo, v.as_ref(), scale, self.spec_iters, self.spec_iom, false);

            // safety factor
            let sf = self.spec_saftey_factor;
            let (a, b, c) = (sf * fit_a, fit_b, sf * fit_c);

            *self = Self {
                a,
                b,
                c,
                spec_norm: spec_norm,
                spec_norm_tol: self.spec_norm_tol,
                spec_iters: self.spec_iters,
                spec_iom: self.spec_iom,
                spec_saftey_factor: self.spec_saftey_factor,
                arnld_q: Some(q),
                arnld_h: Some(h),
                ritz_re: Some(ritz_re),
                ritz_im: Some(ritz_im),
            };
        }
    }

    fn get_bounds(&self) -> (f64, f64, f64) {
        (self.a, self.b, self.c)
    }

    fn get_ritz(&self) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        (self.ritz_re.to_owned(), self.ritz_im.to_owned())
    }

    fn get_shift_scale(&self) -> (f64, f64) {
        let hax1 = (self.b - self.a) / 2.0;
        let hax2 = self.c;
        let shift = (self.a + self.b) / 2.0;
        let scale = (hax1 + hax2) / 2.0;
        (shift, scale)
    }

    fn n_ritz(&self) -> usize {
        match self.ritz_re.as_ref() {
            Some(ritz_re) => { ritz_re.len() },
            _ => 0
        }
    }

    fn get_ritz_leja(&self) -> Option<LejaPoints> {
        let (lp_ritz, _, _) = LejaPoints::new(self.ritz_re.to_owned().unwrap(), self.ritz_im.to_owned().unwrap())
            .normalize(self.a, self.b, self.c)
            .reorder_conj_pairs()
            .rescale(self.a, self.b, self.c);
        Some(lp_ritz)
    }

    fn get_krylov_subspace(&self) -> (Option<Mat<f64>>, Option<Mat<f64>>) {
        (self.arnld_q.to_owned(), self.arnld_h.to_owned())
    }
}

#[derive(Clone, Debug)]
pub struct LejaEllipseAdapterStatic {
    // spectrum parameters
    a: f64,
    b: f64,
    c: f64,
}

impl LejaEllipseAdapterStatic {
    pub fn new(a: f64, b: f64, c: f64) -> Self
    {
        assert!(a <= b);
        assert!(c >= 0.0);
        Self {a, b, c}
    }
}

impl GetSpectrumBounds for LejaEllipseAdapterStatic {
    fn update(&mut self, _a_lo: &dyn LinOp<f64>, _v: MatRef<f64>, _scale: f64) { }

    fn get_bounds(&self) -> (f64, f64, f64) {
        (self.a, self.b, self.c)
    }

    fn get_shift_scale(&self) -> (f64, f64) {
        let hax1 = (self.b - self.a) / 2.0;
        let hax2 = self.c;
        let shift = (self.a + self.b) / 2.0;
        let scale = (hax1 + hax2) / 2.0;
        (shift, scale)
    }
}




/// Using the Gershgorin circle theorem to estimate
/// spectrum paramters.
/// diagonals of matrix are centers of disks
/// sum of each row is radius of each disk
/// take max radius and max diag + radius as spectrum bounds
pub fn spectrum_gershgorin_disks(ext_a_lo: &dyn LinOp<f64>) -> (f64, f64, f64) {
    let mut a: f64 = -1.0;
    let mut b: f64 = 0.0;
    let mut c: f64 = 1.0;

    //let diag = ext_a_lo.inner_lop.apply(eye);
    //let row_sums = ext_a_lo.inner_lop.apply(ones);
    //let a = (diag - row_sums).min();
    //let b = 0.0;
    //let c = row_sums.abs().max();
    todo!("Implement greshgorin disks estimate of spectrum bounds.");
    (a, b, c)
}

/// Using krylov-schur to estimate
/// spectrum paramters.
pub fn spectrum_krylov_schur(
    ext_a_lo: &dyn LinOp<f64>,
    v0: MatRef<f64>,
    scale: f64,
    n: usize,
    tol: f64,
    update_b: bool)
    -> (f64, f64, f64, Vec<f64>, Vec<f64>, Mat<Complex64>)
{
    let nev = std::cmp::min(n, ext_a_lo.nrows());
    let mut eigvals = vec![Complex64::ZERO; nev];
    let mut eigvecs = Mat::<Complex64>::zeros(ext_a_lo.nrows(), nev);
    // let random_f64 = |_| rand::random::<f64>().into();
    // let mut r0: Col<f64> = Col::from_fn(ext_a_lo.nrows(), random_f64);
    // r0 /= r0.norm_l2();
    let r0 = v0.col(0) / v0.norm_l2();

    let par = faer::get_global_parallelism();
    let mut params = faer::matrix_free::eigen::PartialEigenParams::default();
    let stack_req =
        faer::matrix_free::eigen::partial_eigen_scratch(ext_a_lo, nev, par, params);
    let mut membuffer = MemBuffer::new(stack_req);
    let memstack = MemStack::new(&mut membuffer);

    let _partial_eig_info = faer::matrix_free::eigen::partial_eigen(
        eigvecs.rb_mut(),
        &mut eigvals,
        ext_a_lo,
        r0.as_ref(),
        tol,
        par,
        memstack,
        params,
    );

    // extract spectrum parameters from eigenvalues
    let ritz_re = eigvals.iter().map(|v| scale * v.re()).collect::<Vec<f64>>();
    let ritz_im = eigvals.iter().map(|v| scale * v.im()).collect::<Vec<f64>>();
    let a = ritz_re.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let b = ritz_re.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let c = ritz_im.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    if update_b {
        return (*a, *b, *c, ritz_re, ritz_im, eigvecs)
    }
    // let b = 0.0;
    // (*a, b, *c, ritz_re, ritz_im)
    (a.min(-1.0e-2), b.max(0.0), *c, ritz_re, ritz_im, eigvecs)
}

/// Using power iteration to estimate
/// spectrum paramters.
/// WARNING: this method does not work for systems
/// with complex eigenvalues.
pub fn spectrum_pwr_itr(
    ext_a_lo: &dyn LinOp<f64>,
    v0: MatRef<f64>,
    scale: f64,
    n: usize,
    tol: f64)
    -> (f64, f64, f64, Mat<f64>)
{
    let mut b_k = v0.to_owned();
    let mut b_k1 = v0.to_owned();
    let mut eig_old = 1.0e20;
    let mut eig_new = 0.0;
    for _i in 0..n {
        ext_a_lo.apply(
            b_k1.as_mut(),
            b_k.as_ref(),
            faer::get_global_parallelism(),
            MemStack::new(&mut MemBuffer::new(StackReq::empty()))
        );
        let sb_k1 = b_k1.as_ref();
        eig_new = (b_k.transpose() * sb_k1.as_ref())[(0,0)] / (b_k.transpose() * b_k.as_ref())[(0,0)];
        let norm = sb_k1.norm_l2();
        b_k = sb_k1.as_ref() / norm;
        let eig_diff = eig_new - eig_old;
        if eig_diff.abs() < tol {
            break;
        }
        eig_old = eig_new;
    }

    // estimate spetrum parameters from dominant eigv
    let a = (scale * eig_new).min(-1.0e-2);
    let b = 0.0;
    let c = 0.0;
    (a, b, c, b_k)
}

/// Using arnoldi iteration to estimate
/// spectrum paramters.
///
/// #Args
/// * `a_lo` - linear operator, sparse mat or impls method to apply mat to vec
/// * `v0` - initial vector
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth
pub fn spectrum_arnoldi_iom(
    a_lo: &dyn LinOp<f64>,
    v0: MatRef<f64>,
    scale: f64,
    n: usize,
    iom: usize,
    update_b: bool)
    -> (f64, f64, f64, Vec<f64>, Vec<f64>, Mat<f64>, Mat<f64>)
{
    // allocate hessenberg
    let mut hs = faer::Mat::zeros(n+1, n+1);
    let mut qs = faer::Mat::zeros(v0.nrows(), n+1);
    // run arnoldi
    let (_, bdwn_n) = arnoldi_lop_restarted(
        a_lo, scale, v0, hs.as_mut(), qs.as_mut(), 0, n, iom);

    // trim hessenberg to size
    let h_dim = min(n, bdwn_n);
    // get H_{m+1}
    let h = hs.get(0..h_dim+1, 0..h_dim+1).to_owned();
    let q = qs.get(.., 0..h_dim+1).to_owned();
    assert!(h.ncols() == h.nrows());
    assert!(q.ncols() == h.nrows());

    // compute the ritz values of H_{m}
    let ritzv = hs.get(0..h_dim, 0..h_dim).eigenvalues().unwrap();

    // eigenvalues of `dt * a_lo`, matching the time-step
    // used in the Leja polynomial.
    let ritz_re = ritzv.iter().map(|v| v.re()).collect::<Vec<f64>>();
    let ritz_im = ritzv.iter().map(|v| v.im()).collect::<Vec<f64>>();
    let a = ritz_re.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let b = ritz_re.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let c = ritz_im.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    if update_b {
        return (*a, *b, *c, ritz_re, ritz_im, q, h)
    }
    // apply artificial spectrum bounds
    (a.min(-1.0e-2), 0.0, *c, ritz_re, ritz_im, q, h)
}


#[cfg(test)]
mod test_matexp_leja {
    use core::time;
    use std::time::Instant;

    use assert_approx_eq::assert_approx_eq;
    use crate::matexp_krylov::KrylovExpm;
    use crate::mat_utils::mat_mat_approx_eq;
    use crate::matexp_pade::{matexp, phi};
    use crate::test_common::{gen_test_a, gen_test_b, gen_test_c};

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_dd_taylor() {
        // test the divided differences
        let lp = LejaPoints::new_from_lib("leja_circle").slice(0, 100);
        let a = -1.0;
        let b = 0.0;
        let c = 0.5;
        let (lp_sc, shift, scale) = lp.rescale(a, b, c);

        // compute the leja polynomial coeffs
        let coeffs = dd_taylor(&lp_sc, shift, scale, 1.0, 16, 0);

        println!("dd_0: {}", coeffs[0]);
        println!("dd_1: {}", coeffs[1]);
        println!("dd_2: {}", coeffs[2]);
        println!("dd_3: {}", coeffs[3]);
    }

    #[test]
    fn test_spectrum_params() {
        // test the ability of arnoldi procedure to produce
        // correct spectrum parameters with a matrix with known
        // eigenvalues.

        // Generate a test 3x3 matrix
        let (test_a, test_v) = gen_test_a();
        let (test_b, _test_v) = gen_test_b();

        // compute the spectrum parameters with arnoldi with incomplete orthogonalization
        let (a, b, c, _, _, _, _) = spectrum_arnoldi_iom(&test_a.as_ref(), test_v.as_ref(), 1.0, 10, 2, true);
        println!("Spectrum params: a= {a}, b= {b}, c= {c}");
        assert_approx_eq!(a, -1.0, 1e-1);
        assert_approx_eq!(b, -1.0e-3, 1e-1);
        assert_approx_eq!(c,  0.0, 1e-1);

        // build an extended linear operator
        let mut vbk: Vec<MatRef<f64>> = vec![];
        vbk.push(test_v.as_ref());
        let ext_a_lo = DynRefExtendedLinOp::new(1.0, &test_a, &vbk);
        let (ext_a, ext_b, ext_c, _, _, _, _) = spectrum_arnoldi_iom(&ext_a_lo, test_v.as_ref(), 1.0, 10, 10, true);

        // check for consistency
        assert_approx_eq!(a, ext_a, 1e-1);
        // assert_approx_eq!(b, ext_b);
        assert_approx_eq!(c, ext_c, 1e-1);

        // run power iteration
        let (pwr_a, _pwr_b, _pwr_c, _) = spectrum_pwr_itr(&ext_a_lo, test_v.as_ref(), 1.0, 40, 1e-5);
        // check for consistency
        assert_approx_eq!(a, pwr_a, 1e-1);

        // check spectrum parameters of matrix with conj complex eig pair
        let (a, b, c, _, _, _, _) = spectrum_arnoldi_iom(&test_b.as_ref(), test_v.as_ref(), 1.0, 10, 2, true);
        println!("Spectrum params: a= {a}, b= {b}, c= {c}");
        // eigen decomp of b
        let b_eigs = test_b.eigenvalues().unwrap();
        let b_eigs_re: Vec<f64> = b_eigs.iter().map(|x| { x.re() }).collect();
        let b_eigs_im: Vec<f64> = b_eigs.iter().map(|x| { x.im() }).collect();
        let min_b_re = b_eigs_re.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let max_b_re = b_eigs_re.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        let max_b_im = b_eigs_im.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    }

    #[test]
    fn test_taylor_expmv() {
        // test that exp(dt*A)*v products can be computed by a
        // taylor polynomial method.

        // Generate a test 3x3 matrix
        let (test_a, test_v) = gen_test_a();

        // compute the matrix matexp(dt*A)*v using dense impl
        let expm_tay = phik_taylor(test_a.as_ref(), 0.0, 1.0, 16, 0);
        let expmv_tay_dense = expm_tay.as_ref() * test_v.as_ref();

        // compute the matrix matexp(dt*A)*v using matfree impl
        let lp = LejaPoints::new(vec![], vec![]);
        let leja_ellipse_adapter = LejaEllipseAdapterStatic::new(0., 1.0, 0.);
        let leja_phikv_eval = LejaPhiEval::new(
            lp, 20, 1e-8, "taylor", "dd_phi", false, Box::new(leja_ellipse_adapter));
        let mut expmv_tay_pm = faer::Mat::zeros(test_a.nrows(), 1);
        leja_phikv_eval.taylor_expmv(expmv_tay_pm.as_mut(),
            &test_a, 1.0, test_v.as_ref(), 0.0, 1.0, 20);
        println!("{:?}", expmv_tay_dense.as_ref());
        println!("{:?}", expmv_tay_pm.as_ref());

        // Ensure results are consistent.
        mat_mat_approx_eq(
            expmv_tay_pm.as_ref(), expmv_tay_dense.as_ref(), 1e-8);

        // compute the matrix phi_2(dt*A)*v using dense impl
        let phi2v_tay = phik_taylor((1.0*&test_a).as_ref(), 0.0, 1.0, 16, 2) * test_v.as_ref();
        let phi2v_pade = phi((1.0*&test_a).as_ref(), 2) * test_v.as_ref();
        mat_mat_approx_eq(
            phi2v_pade.as_ref(), phi2v_tay.as_ref(), 1e-8);
    }

    #[test]
    fn test_leja_expmv() {
        // test that exp(dt*A)*v products can be computed by a
        // leja polynomial method.
        // load leja points
        let lp_re = LejaPoints::new_from_lib("leja_real").slice(0, 100);
        let lp_clp = LejaPoints::new_from_lib("leja_circle").slice(0, 100);
        assert!(lp_re.n_leja() == lp_clp.n_leja());
        assert!(lp_re.n_leja_real() == lp_re.n_leja());
        assert!(lp_clp.n_leja_real() == 2);
        let lp = lp_clp;

        // Generate a test 3x3 matricies
        let (test_a, test_v) = gen_test_a();
        let (test_b, _) = gen_test_b();
        let test_mats = vec![test_a, test_b];

        for test_m in test_mats.iter() {
            // compute the matexp(dt*A)*v product via leja poly approx
            let leja_ellipse_adapter = LejaEllipseAdapterArnoldiIOM::new(-1.0, 0.0, 0.0, 1e-8, 10, 2, 1.0);
            let mut leja_phikv_eval = LejaPhiEval::new(
                lp.clone(), 80, 1e-8, "clapm", "dd_taylor", false, Box::new(leja_ellipse_adapter));
            leja_phikv_eval.apply_prepare(&test_m, 1.0, test_v.as_ref(), 0);
            let expmv_leja_pm = leja_phikv_eval.apply_phi_k(&test_m, 1.0, test_v.as_ref(), 0);

            // Ensure results are consistent with pade methods.
            let expmv_pade_dense = matexp(test_m.as_ref(), 1.0) * test_v.as_ref();
            println!("leja expmv: {:?}", &expmv_leja_pm);
            println!("pade expmv: {:?}", &expmv_pade_dense);
            mat_mat_approx_eq(
                expmv_leja_pm.as_ref(), expmv_pade_dense.as_ref(), 1e-8);
        }
    }

    #[test]
    fn test_leja_phikv_simple() {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // leja polynomial method.

        // load leja points
        let lp = LejaPoints::new_from_fn("leja_circle").slice(0, 100);

        // Generate a test matrix
        let (test_b, test_v) = gen_test_b();

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref(),];

        // setup the phi evaluator
        let leja_ellipse_adapter = LejaEllipseAdapterArnoldiIOM::new(-1.0, 0.0, 0.0, 1e-8, 10, 2, 1.0);
        let mut leja_phikv_eval = LejaPhiEval::new(
            lp, 80, 1e-8, "clapm", "dd_taylor", false, Box::new(leja_ellipse_adapter));

        // compute phi_0(dt*A)*b0
        // fn apply_phi_k_v(&self, a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64> {
        let dt = 1.0;
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        leja_phikv_eval.apply_prepare(&test_b, dt, test_v.as_ref(), 0);
        let phi0mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, dt, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi0mv_pade_dense = matexp(test_b.as_ref(), 1.0) * test_v.as_ref();
        println!("leja phi0mv: {:?}", &phi0mv_leja_pm);
        println!("pade phi0mv: {:?}", &phi0mv_pade_dense);
        mat_mat_approx_eq(
            phi0mv_leja_pm.as_ref(), phi0mv_pade_dense.as_ref(), 1e-8);

        // generate vb vector: vb = [b0, b1, ... bk]
        let zeros = faer::Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), test_v.as_ref()];

        // compute phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi1mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, dt, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi1mv_pade_dense = phi(test_b.as_ref(), 1) * test_v.as_ref();
        println!("leja phi1mv: {:?}", &phi1mv_leja_pm);
        println!("pade phi1mv: {:?}", &phi1mv_pade_dense);
        mat_mat_approx_eq(
            phi1mv_leja_pm.as_ref(), phi1mv_pade_dense.as_ref(), 1e-8);
    }

    fn _test_leja_ritz_phikv(
        dt: f64, test_b: Mat<f64>, test_v: Mat<f64>, krylov_reuse: bool,
        max_arnoldi_iters: usize, max_substeps: usize)
    {
        // load leja points
        let lp = LejaPoints::new_from_fn("leja_circle").slice(0, 300);

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref(),];

        // setup the phi evaluator
        let iom = 4;
        let leja_ellipse_adapter = LejaEllipseAdapterArnoldiIOM::new(
            -1.0, 0.0, 0.0, 1e-8, max_arnoldi_iters, iom, 1.0);
        let mut leja_phikv_eval = LejaPhiEval::new(
            lp, 280, 1e-15, "clapm", "dd_phi", krylov_reuse, Box::new(leja_ellipse_adapter));

        // print the ritz values
        let (_a, _b, _c, ritz_re, ritz_im, q, h) = spectrum_arnoldi_iom(
            &test_b.as_ref(), test_v.as_ref(), dt, max_arnoldi_iters, iom, false);
        println!("ritz re: {:?}", ritz_re);
        println!("ritz im: {:?}", ritz_im);

        // compute phi_0(dt*A)*b0
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        leja_phikv_eval.apply_prepare(&test_b, dt, test_v.as_ref(), 0);
        leja_phikv_eval.set_max_substeps(max_substeps);
        let phi0mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi0mv_pade_dense = matexp(test_b.as_ref(), dt) * test_v.as_ref();
        println!("leja_ritz phi0mv: {:?}", &phi0mv_leja_pm);
        println!("pade phi0mv: {:?}", &phi0mv_pade_dense);
        mat_mat_approx_eq(
            phi0mv_leja_pm.as_ref(), phi0mv_pade_dense.as_ref(), 1e-7);

        // generate vb vector: vb = [b0, b1, ... bk]
        let zeros = faer::Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), test_v.as_ref()];

        // compute phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        leja_phikv_eval.apply_prepare(&test_b, dt, test_v.as_ref(), 1);
        leja_phikv_eval.set_max_substeps(max_substeps);
        let phi1mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi1mv_pade_dense = phi((dt*test_b.as_ref()).as_ref(), 1) * test_v.as_ref();
        println!("leja_ritz phi1mv: {:?}", &phi1mv_leja_pm);
        println!("pade phi1mv: {:?}", &phi1mv_pade_dense);
        mat_mat_approx_eq(
            phi1mv_leja_pm.as_ref(), phi1mv_pade_dense.as_ref(), 1e-7);

        // generate vb vector: vb = [b0, b1, ... bk]
        let zeros = faer::Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), zeros.as_ref(), test_v.as_ref()];

        // compute phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        leja_phikv_eval.apply_prepare(&test_b, dt, test_v.as_ref(), 2);
        leja_phikv_eval.set_max_substeps(max_substeps);
        let phi2mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, 1.0, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi2mv_pade_dense = phi((dt*test_b.as_ref()).as_ref(), 2) * test_v.as_ref();
        println!("leja_ritz phi2mv: {:?}", &phi2mv_leja_pm);
        println!("pade phi2mv: {:?}", &phi2mv_pade_dense);
        mat_mat_approx_eq(
            phi2mv_leja_pm.as_ref(), phi2mv_pade_dense.as_ref(), 1e-7);
    }

    #[test]
    fn test_leja_phikv_small_krylov_noreuse() {
        let dt = 1.0;
        let (test_b, test_v) = gen_test_b();
        _test_leja_ritz_phikv(dt, test_b, test_v, false, 10, 0);
    }

    #[test]
    fn test_leja_phikv_small_krylov_reuse() {
        let dt = 1.0;
        let (test_b, test_v) = gen_test_b();
        _test_leja_ritz_phikv(dt, test_b, test_v, true, 10, 0);
    }

    #[test]
    fn test_leja_phikv_large_krylov_noreuse() {
        // similar test on a larger system
        let dt = 1.2;
        //let (test_b, test_v) = gen_test_c(80);
        //_test_leja_ritz_phikv(dt, 2.0*test_b, test_v, false, 20);
        let (test_b, test_v) = gen_test_c(40);
        _test_leja_ritz_phikv(dt, 1.8*test_b, test_v, false, 10, 0);
    }

    #[test]
    fn test_leja_phikv_large_krylov_reuse() {
        // similar test on a larger system
        let dt = 1.2;
        //let (test_b, test_v) = gen_test_c(80);
        //_test_leja_ritz_phikv(dt, 2.0*test_b, test_v, true, 20);
        let (test_b, test_v) = gen_test_c(40);
        _test_leja_ritz_phikv(dt, 1.8*test_b, test_v, true, 10, 0);
    }

    #[test]
    fn test_leja_phikv_large_krylov_reuse_substep() {
        // similar test on a larger system
        let dt = 1.2;
        let (test_b, test_v) = gen_test_c(40);
        _test_leja_ritz_phikv(dt, 1.8*test_b, test_v, true, 20, 4);
    }

    #[test]
    fn test_leja_phikv_krylov_reuse_substep_correct() {
        let dt = 1.2;
        for max_substeps in [1_usize, 2, 4, 8] {
            println!("=== krylov reuse substep test, max_substeps={max_substeps} ===");
            let (test_b, test_v) = gen_test_c(40);
            _test_leja_ritz_phikv(dt, 1.8*test_b, test_v, true, 20, max_substeps);
        }
    }

    fn _test_leja_phikv_sincos(max_substeps: usize) {
        // check exp(dt*A)*v for system with pure imaginary eigenvalues
        // A = [[0, -1], [1, 0]]
        // with initial conditions v0=[1, 0]
        // where the analytic soution is
        // v_(t) = [-cos(t), sin(t)]
        let dt = 1.2;
        let tf: f64 = 1.*dt;
        let lambda_a = 0.0;
        let lambda_b = 1.0;
        let test_a = faer::mat![
            [lambda_a, lambda_b],
            [-lambda_b, lambda_a]
            ];
        // Generate a test vector
        let test_v = faer::mat![
            [-1.0],
            [0.0],
            ];

        // load leja points
        let lp = LejaPoints::new_from_lib("leja_circle").slice(0, 100);
        // setup the phi evaluator
        let krylov_reuse = false;
        let max_order = 50;
        let leja_tol = 1.0e-8;
        let leja_a = -1.0e-18;
        let leja_b = 0.0;
        let leja_c = 1.0;
        // let leja_a = -1.0e-12;
        // let leja_c = 1.0e-12;

        let leja_ellipse_adapter = LejaEllipseAdapterStatic::new(leja_a, leja_b, leja_c);
        let mut leja_phikv_eval = LejaPhiEval::new(
            lp, max_order, leja_tol, "clapm", "dd_taylor",
            krylov_reuse, Box::new(leja_ellipse_adapter));
        leja_phikv_eval.set_max_substeps(max_substeps);
        assert_eq!(leja_phikv_eval.max_substeps, max_substeps);

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref(),];
        // compute phi_0(dt*A)*v0
        let ext_a_lo = DynRefExtendedLinOp::new(dt, &test_a, &test_vb);
        leja_phikv_eval.apply_prepare(&test_a, 1.0, test_v.as_ref(), 0);
        let phi0_v0: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_a_lo, 1.0, &test_vb);

        println!("dt: {:}", dt);
        println!("Leja phi0(dt*A)*v0: {:?}", phi0_v0);

        // compare to analytic result
        assert_approx_eq!(phi0_v0[(0, 0)], -f64::cos(tf), 1e-8);
        assert_approx_eq!(phi0_v0[(1, 0)], f64::sin(tf), 1e-8);
    }

    #[test]
    fn test_leja_phikv_sincos_nosubstep() {
        // Test leja evaluator with no substepping
        _test_leja_phikv_sincos(0);
    }

    #[test]
    fn test_leja_phikv_sincos_substep() {
        // Test leja evaluator with substeps
        _test_leja_phikv_sincos(4);
    }

    fn _test_dd_phi(a: f64, b: f64, c: f64, h: f64, n: usize, tol: f64) {
        // Verify dd_phi agrees with dd_taylor
        let lp = LejaPoints::new_from_lib("leja_circle").slice(0, n);
        let (lp_sc, shift, scale) = lp.rescale(a, b, c);
        let k = 0;

        let start = Instant::now();
        let coeffs_ts  = dd_taylor(&lp_sc, shift, scale, 1.0, 16, k);
        let coeffs_ts_time = start.elapsed().as_secs_f64();
        // Paper recommends >= 30 extra Taylor terms; use 30 here.
        let start = Instant::now();
        let coeffs_phi = dd_phi(&lp_sc, shift, scale, h, 32, k);
        let coeffs_phi_time = start.elapsed().as_secs_f64();
        println!("k={k}: dd_taylor[0]={}, dd_phi[0]={}",
                 coeffs_ts[0], coeffs_phi[0]);
        println!("k={k}: dd_taylor[10]={:0.6e}, dd_phi[10]={:0.6e}",
                 coeffs_ts[10], coeffs_phi[10]);
        println!("k={k}: dd_taylor[20]={:0.6e}, dd_phi[20]={:0.6e}",
                 coeffs_ts[20], coeffs_phi[20]);
        if n > 200 {
            println!("k={k}: dd_taylor[200]={:0.6e}, dd_phi[200]={:0.6e}",
                     coeffs_ts[200], coeffs_phi[200]);
            // check that at large sequence sizes the methods agree to within 20% rel tol
            assert_approx_eq!((coeffs_phi[200].re - coeffs_ts[200].re).abs() / coeffs_ts[200].re.abs(), 0.0, 0.2);
        }
        if n > 250 {
            println!("k={k}: dd_taylor[250]={:0.6e}, dd_phi[250]={:0.6e}",
                     coeffs_ts[250], coeffs_phi[250]);
        }
        println!("n_leja: {n}, dd_taylor time: {coeffs_ts_time} (s)");
        println!("n_leja: {n}, dd_phi time: {coeffs_phi_time} (s).");

        for i in 0..lp_sc.n_leja() {
            assert_approx_eq!(coeffs_ts[i].re, coeffs_phi[i].re, tol);
            assert_approx_eq!(coeffs_ts[i].im, coeffs_phi[i].im, tol);
        }
    }

    #[test]
    fn test_dd_phi() {
        // Test dd_phi method for a large ellipse.
        let mut a = -508.2;
        let mut b = 0.001;
        let mut c = 50.58;
        _test_dd_phi(a, b, c, 1.0, 60,  1e-10);
        _test_dd_phi(a, b, c, 1.0, 60,  1e-10);
        _test_dd_phi(a, b, c, 1.0, 60,  1e-10);
        _test_dd_phi(a, b, c, 1.0, 100, 1e-10);
        _test_dd_phi(a, b, c, 1.0, 300, 1e-10);

        // Test dd_phi method for a small ellipse
        a = -1.2;
        b = 0.0;
        c = 0.58;
        _test_dd_phi(a, b, c, 1.0, 60,  1e-10);
        _test_dd_phi(a, b, c, 1.0, 100, 1e-10);
        _test_dd_phi(a, b, c, 1.0, 60,  1e-10);
        _test_dd_phi(a, b, c, 1.0, 100, 1e-10);
    }

    #[test]
    fn test_leja_circle() {
        let lp_fn = LejaPoints::new_from_fn("leja_circle").slice(0, 50);
        let lp_lib = LejaPoints::new_from_lib("leja_circle").slice(0, 50);
        for (lp_fn_re, lp_lib_re) in lp_fn.leja_re.iter().zip(lp_lib.leja_re.iter()) {
            println!("{lp_lib_re}, {lp_fn_re}");
            // assert_approx_eq!(lp_fn_re, lp_lib_re);
        }
        for (lp_fn_im, lp_lib_im) in lp_fn.leja_im.iter().zip(lp_lib.leja_im.iter()) {
            // assert_approx_eq!(lp_fn_im, lp_lib_im);
        }
    }

    #[test]
    fn test_leja_line() {
        let lp_fn = LejaPoints::new_from_fn("leja_real").slice(0, 50);
        let lp_lib = LejaPoints::new_from_lib("leja_real").slice(0, 50);
        for (lp_fn_re, lp_lib_re) in lp_fn.leja_re.iter().zip(lp_lib.leja_re.iter()) {
            println!("{lp_lib_re}, {lp_fn_re}");
            assert_approx_eq!(lp_fn_re, lp_lib_re);
        }
        for (lp_fn_im, lp_lib_im) in lp_fn.leja_im.iter().zip(lp_lib.leja_im.iter()) {
            assert_approx_eq!(lp_fn_im, lp_lib_im);
        }
    }
}
