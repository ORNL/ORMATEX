use faer::complex::Complex64;
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
/// Leja point exp(A*dt)*v methods
///
use faer::reborrow::*;
use faer::prelude::*;
use faer::matrix_free::LinOp;
use faer::complex::ComplexFloat;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};

use std::cmp::{max};
use statrs::function::{factorial};
use csv;

use crate::matexp_pade;
use crate::ode_sys::{DynRefExtendedLinOp};
use crate::matexp_traits::{LinOpPhikvEvaluator};
use crate::arnoldi::{arnoldi_lop};

/// Pre-generated Leja points from file
/// Real leja points in [-2, 2]
const LEJA_REAL_CSV: &str = std::include_str!("leja_points_real");
/// Complex conjugate leja points are on the unit circle.
const LEJA_CIRCLE_CSV: &str = std::include_str!("leja_points_circle");


/// Rescale the leja points to bound the interval [a, b, -c, +c].
///
/// Returns:
///         (leja_re, leja_im, shift, scale)
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
/// Returns:
///         (leja_re, leja_im)
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

    /// Build the matrix Xi =
    ///         [[leja_00,       0,       0],
    ///         [[      1, leja_11,       0],
    ///         [[      0,       1, leja_22],
    ///         [[      ...                ]]
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
            // leja_reordered.push(c64::new(self.leja_re[*im_i], -self.leja_im[*im_i]));
        }
        for lp in leja_reordered.iter() {
            leja_re_reordered.push((*lp).re());
            leja_im_reordered.push((*lp).im());
        }
        // create reordered leja points
        Self::new(leja_re_reordered, leja_im_reordered)
    }

    /// Normalize the Leja points such that the capacity is 1
    pub fn normalize(&self, a: f64, b: f64, c: f64) -> Self {
        let (leja_re_normed, leja_im_normed) = inv_shift_scale_leja(
            self.leja_re.as_ref(), self.leja_im.as_ref(), a, b, c, 1.0, 1.0);
        Self::new_from_col(leja_re_normed, leja_im_normed)
    }

    /// Extract a slice form the original Leja sequence
    pub fn slice(&self, a: usize, b: usize) -> Self {
        assert!(a <= b);
        let b_end = std::cmp::min(b, self.n_leja());
        Self::new_from_col(self.leja_re.get(a..b_end).to_owned(), self.leja_im.get(a..b_end).to_owned())
    }

    /// Rescale the leja points
    pub fn rescale(&self, a: f64, b: f64, c: f64) -> (Self, f64, f64) {
        let (leja_sc_re, leja_sc_im, shift, scale) = shift_scale_leja(
            self.leja_re.as_ref(), self.leja_im.as_ref(), a, b, c);
        (Self::new_from_col(leja_sc_re, leja_sc_im), shift, scale)
    }
}

/// computes the factorial
pub fn ufactorial(num: usize) -> f64 {
    (1..=num).product::<usize>() as f64
}

/// Compute the dense matrix exponential using tayler series
///
/// # Args
/// * `A` : the matrix
/// * `shift` : spectrum shift parameter. 0.0 for unshifted matexp.
/// * `scale` : spectrum shift parameter. 1.0 for unscaled matexp.
/// * `p` : polynomial order
///
pub fn expm_taylor<T: faer::traits::ComplexField>(A: MatRef<T>, shift: f64, scale: f64, p: usize) -> Mat<T>
{
    let mut M: Mat<T> = scale * A.as_ref();
    let mut ts_expm: Mat<T> = faer::Mat::identity(M.nrows(), M.ncols());
    for i in 0..p {
        ts_expm += M.as_ref() / factorial::factorial((i+1) as u64);
        M = A.as_ref() * M.as_ref();
    }
    shift.exp() * ts_expm
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
///
pub fn dd_expm_taylor(leja_x: &LejaPoints, shift: f64, scale: f64, h: f64, p: usize) -> Col<c64>
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
    let s_scale = z.norm_l1();
    let s = max((s_scale.ln() / (3.0 as f64).ln()).ceil() as i32, 1);
    let hs = 1.0 / (2.0 as f64).powi(s);

    // compute expm(Z)
    let mut f_out = expm_taylor((hs*h*z).as_ref(), 0.0, 1.0, p);

    // squaring
    for _i in 0..s {
        f_out = f_out.as_ref() * f_out.as_ref();
    }

    // reshift and extract first col
    faer::Scale( (h * mu).exp() ) * f_out.col(0)
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
    n_leja_real: usize,
    tol: f64,
    abort_tol: f64,
    shift: f64,
    scale: f64,
    max_substeps: usize,
    krylov_reuse: bool,
    spec_norm: f64,
    spec_norm_tol: f64,
    spec_iters: usize,
    spec_method: String,
    arnld_q: Option<Mat<f64>>,
    arnld_h: Option<Mat<f64>>,
    ritz_re: Option<Vec<f64>>,
    ritz_im: Option<Vec<f64>>,
}


/// Leja point phi function evaluator
impl LejaPhiEval {
    /// Create a new leja point phi function evaluator
    ///
    /// #Args
    /// * `leja_x` - the leja points
    /// * `m` - maximum polynomial approximation order
    /// * `shift` - the leja point shift parameter
    /// * `scale` - the leja point scale parameter
    /// * `tol` - leja polynomial approximation tolerance
    /// * `spec_norm_tol` - tolerance used to trigger recomputation of spectrum parameters
    /// * `spec_iters` - maximum number of arnoldi iterations used in spectrum parameter calc
    /// * `spec_method` - method used to estimate spectrum parameters
    pub fn new(
        leja_x: LejaPoints,
        m: usize,
        shift: f64,
        scale: f64,
        tol: f64,
        spec_norm_tol: f64,
        spec_iters: usize,
        spec_method: &str,
        krylov_reuse: bool) -> Self
    {
        Self {
            m: m,
            n_leja_real: (&leja_x).n_leja_real(),
            leja_x: leja_x.clone(),
            leja_base: leja_x,
            tol: tol,
            abort_tol: 1e26,
            shift: shift,
            scale: scale,
            max_substeps: 0,
            krylov_reuse: krylov_reuse,
            spec_norm: -1.0,
            spec_norm_tol: spec_norm_tol,
            spec_iters: spec_iters,
            spec_method: spec_method.to_string(),
            arnld_q: None,
            arnld_h: None,
            ritz_re: None,
            ritz_im: None,
        }
    }

    /// Create a new leja point phi function evaluator
    ///
    /// #Args
    /// * `leja_x` - the leja points
    /// * `m` - maximum polynomial approximation order
    /// * `a` - the minimum real eigenvalue spectrum parameter
    /// * `b` - the maximum real eigenvalue spectrum parameter
    /// * `c` - the maximum imaginary eigenvalue spectrum parameter
    /// * `tol` - leja polynomial approximation tolerance
    /// * `spec_norm_tol` - tolerance used to trigger recomputation of spectrum parameters
    /// * `spec_iters` - maximum number of arnoldi iterations used in spectrum parameter calc
    pub fn new_from_abc(
        leja_x: LejaPoints,
        m: usize,
        a: f64,
        b: f64,
        c: f64,
        tol: f64,
        spec_norm_tol: f64,
        spec_iters: usize,
        spec_method: &str,
        krylov_reuse: bool) -> Self
    {
        let (lp, shift, scale) = leja_x.rescale(a, b, c);
        Self {
            m: m,
            n_leja_real: (&lp).n_leja_real(),
            leja_x: lp,
            leja_base: leja_x,
            tol: tol,
            abort_tol: 1e10,
            shift: shift,
            scale: scale,
            max_substeps: 0,
            krylov_reuse: krylov_reuse,
            spec_norm: -1.0,
            spec_norm_tol: spec_norm_tol,
            spec_iters: spec_iters,
            spec_method: spec_method.to_string(),
            arnld_q: None,
            arnld_h: None,
            ritz_re: None,
            ritz_im: None,
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
    /// Computes the matrix exponential-vector product: exp(dt*A)*u
    ///
    /// #Args
    /// * `pm` - the output vector holding the polynomial approximation of
    ///    the matrix exponential-vector product.
    /// * `ext_a_lo` - the linear operator A
    /// * `dt` - the stepsize
    /// * `u` - the rhs vector
    /// * `shift` - the leja point sequence shift
    /// * `scale` - the leja point sequence scale
    /// * `coeffs` - the leja polynomial coefficients
    fn real_leja_expmv<T: LinOp<f64>>(
        &self,
        mut pm: MatMut<f64>,
        ext_a_lo: &T,
        dt: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        coeffs: ColRef<c64>,
        ) -> (bool, usize)
    {
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

        // compute leja poly and check for convergence each iter
        for i in 1..self.m {
            if converged {
                break;
            }
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                par,
                MemStack::new(&mut mem_buf)
                );
            vm = (dt * av.as_ref() - leja_x_sc[i-1]*vm) / scale;
            // leja polynomial update
            // pm.copy_from( pm.as_ref() + coeffs[i].re * vm.as_ref() );
            pm += coeffs[i].re * vm.as_ref();

            // check error estimate
            err_est = (coeffs[i].re * vm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
            if err_est > self.abort_tol {
                println!("Hit abort tol: {err_est:0.2e}. Consider a smaller time step size.");
                break;
            }
        }
        println!("ReLPM time (s): {}", clock.elapsed().as_secs_f64());
        (converged, iter)
    }

    /// Taylor series method to estimate the action of
    /// the matrix exponential on a vector.
    ///
    /// #Args
    /// * `pm` - the output vector holding the polynomial approximation of
    ///    the matrix exponential-vector product.
    /// * `ext_a_lo` - the linear operator A
    /// * `dt` - the stepsize
    /// * `u` - the rhs vector
    /// * `shift` - location on the real axis about which the
    ///    taylor expansion is conducted.
    /// * `scale` - unused
    fn taylor_expmv<T: LinOp<f64>>(
        &self,
        mut pm: MatMut<f64>,
        ext_a_lo: &T,
        dt: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        ) -> (bool, usize)
    {
        let clock = std::time::Instant::now();
        let mut iter: usize = 0;
        let norm_u: f64 = u.norm_l2();
        let mut err_est = 2. * norm_u;
        let mut converged: bool = err_est == 0.;

        let mut av = u.to_owned();
        let mut vm = u.to_owned();

        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(ext_a_lo.apply_scratch(u.ncols(), par));

        // compute first term of the taylor polynomial
        // ext_a_lo.apply(pm, u, par, mem_scratch);
        let mut coeff = 1.0;
        pm.copy_from(coeff * u);

        for j in 1..self.m {
            if converged {
                break;
            }
            coeff = 1.0 / factorial::factorial(j as u64);
            let mem_scratch = MemStack::new(&mut mem_buf);
            ext_a_lo.apply(av.as_mut(), vm.as_ref(), par, mem_scratch);
            vm = dt * av.as_ref();
            pm += coeff * vm.as_ref();

            // check error estimate
            err_est = (coeff * vm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;

        }
        println!("TS time (s): {}", clock.elapsed().as_secs_f64());
        (converged, iter)
    }

    /// Compute the augmenting first term in the krylov-leja sequence
    /// with krylov subspace polynomial, if available.
    ///
    /// If the krylov subspace has not been computed, or is unavailable,
    /// this routine returns None.
    ///
    /// Ref:
    /// Caliari, Marco, Fabio Cassini, and Franco Zivcovich.
    /// "BAMPHI: Matrix-free and transpose-free action of linear combinations
    /// of phi-functions from exponential integrators."
    /// Journal of Computational and Applied Mathematics 423 (2023): 114973.
    ///
    fn krylov_poly_expmv(
        &self,
        leja_x_sc_re: ColRef<f64>,
        leja_x_sc_im: ColRef<f64>,
        coeffs: ColRef<c64>,
        norm_u: f64
        )
        -> Result<(usize, Mat<f64>, Mat<f64>), ()>
    {
        match (&self.arnld_q, &self.arnld_h, &self.ritz_re, &self.ritz_im) {
            (Some(q), Some(h), Some(ritz_re), Some(ritz_im)) => {
                // number of ritz values available
                let n_r = ritz_re.len();

                // convert to complex for interpolation at the (complex-conj) ritz values
                let cmplx_h: Mat<c64> = faer::Mat::from_fn(
                    h.nrows(), h.ncols(), |i, j| { c64::new(h[(i, j)], 0.0) } );
                let mut dr: Mat<c64> = Mat::zeros(h.nrows(), 1);
                dr[(0, 0)] = c64::new(1.0, 0.0);
                let gamma = c64::new(self.scale, 0.0);

                // compute the first n_r polynomial terms
                let mut xi = faer::Scale(coeffs[0]) * dr.as_ref();
                for r in 1..=n_r {
                    println!("{r}, krylov pre lp: {:0.8} + {:0.8}i, dd: {:0.6e}", leja_x_sc_re[r-1], leja_x_sc_im[r-1], coeffs[r]);
                    let z = c64::new(leja_x_sc_re[r-1], leja_x_sc_im[r-1]);
                    dr = (cmplx_h.as_ref()*dr.as_ref() - faer::Scale(z)*dr.as_ref()) / faer::Scale(gamma);
                    xi += faer::Scale(coeffs[r]) * dr.as_ref();
                }
                // convert to reals
                let xi_re = faer::Mat::from_fn(
                    xi.nrows(), xi.ncols(), |i, j| { xi[(i, j)].re } );
                let mut dr_re = faer::Mat::from_fn(
                    dr.nrows(), dr.ncols(), |i, j| { dr[(i, j)].re } );
                let xr_re = norm_u * q.as_ref() * xi_re;
                dr_re = norm_u * q.as_ref() * dr_re;

                // println!("n_r: {n_r}, krylov_xr: {:?}", xr_re.as_ref());
                // println!("n_r: {n_r}, krylov_dr: {:?}", dr_re.as_ref());
                Ok((n_r, xr_re, dr_re))
            },
            _ => Err(())
        }
    }

    /// Complex conjugate leja point method (CLaPM).
    ///
    /// Computes the matrix exponential-vector product: exp(dt*A)*u
    ///
    /// #Args
    /// * `pm` - the output vector holding the polynomial approximation of
    ///    the matrix exponential-vector product.
    /// * `ext_a_lo` - the linear operator A
    /// * `dt` - the stepsize
    /// * `u` - the rhs vector
    /// * `shift` - the leja point sequence shift
    /// * `scale` - the leja point sequence scale
    /// * `coeffs` - the leja polynomial coefficients
    fn complex_conj_leja_expmv<T: LinOp<f64>>(
        &self, mut pm: MatMut<f64>,
        ext_a_lo: &T,
        dt: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        coeffs: ColRef<c64>,
        ) -> (bool, usize)
    {
        let mut iter: usize = 0;
        let norm_u: f64 = u.norm_l2();
        let mut err_est = 2. * norm_u;
        let mut converged: bool = err_est == 0.;

        // shift and scale leja points to align to the spectrum parameters
        let (leja_x_sc_re, leja_x_sc_im) = self.leja_x.leja_sc(shift, scale);

        // use the real leja point method if leja points are on the real line
        if self.n_leja_real >= self.m {
            // use taylor series if leja points are all near 0
            if self.scale.abs() < 1.0e-20 {
                return self.taylor_expmv(pm, ext_a_lo, dt, u, shift, scale)
            }
            else {
                return self.real_leja_expmv(
                    pm, ext_a_lo, dt, u, shift, scale, coeffs)
            }
        }

        let clock = std::time::Instant::now();
        // first term of leja polynomial
        pm.copy_from( coeffs[0].re * u );
        let mut vm = u.to_owned();
        let mut av = u.to_owned();
        let mut aq = u.to_owned();

        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(ext_a_lo.apply_scratch(u.ncols(), par));

        // Augment leja sequence with krylov subspace polynomial if available
        let krylov_res = self.krylov_poly_expmv(
            leja_x_sc_re.as_ref(), leja_x_sc_im.as_ref(), coeffs, norm_u);
        let mut r: usize = 0;  // number of ritz values
        match krylov_res {
            Ok((n_r, xr, dr)) => {
                pm.copy_from(xr);
                vm = dr;
                r = n_r;
            }
            _ => {}
        }

        // extract next m>r leja points in the sequence
        let n_leja_real = self.leja_x.slice(r, r+10).n_leja_real();
        assert!(n_leja_real == 2);
        println!("n_leja_real: {n_leja_real}");

        // compute leja polynomial terms for leading real points
        for i in 1+r..=n_leja_real+r {
            if converged {
                break;
            }
            println!("{i}, clapm real lp: {:0.8} + {:0.8}i, dd: {:0.6e}", leja_x_sc_re[i-1], leja_x_sc_im[i-1], coeffs[i]);
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                par,
                MemStack::new(&mut mem_buf)
                );
            vm = (dt * av.as_ref() - leja_x_sc_re[i-1]*vm) / scale;
            // leja polynomial update
            // pm.copy_from( pm.as_ref() + coeffs[i].re * vm.as_ref() );
            pm += coeffs[i].re * vm.as_ref();

            // check error estimate
            err_est = (coeffs[i].re * vm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
        }

        // compute remaining leja polynomial terms suported at
        // conjugate complex points.
        for i in (n_leja_real+1+r..self.m).step_by(2) {
            if converged {
                break;
            }
            println!("{}, clapm conj lp: {:0.8} + {:0.8}i, dd: {:0.6e}", i, leja_x_sc_re[i-1], leja_x_sc_im[i-1], coeffs[i]);
            println!("{}, clapm conj lp: {:0.8} + {:0.8}i, dd: {:0.6e}", i+1, leja_x_sc_re[i], leja_x_sc_im[i], coeffs[i+1]);
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                par,
                MemStack::new(&mut mem_buf)
                );
            let qm = (dt * av.as_ref() - leja_x_sc_re[i-1]*vm.as_ref()) / scale;
            pm += coeffs[i].re * qm.as_ref();

            ext_a_lo.apply(aq.as_mut(), qm.as_ref(),
                par,
                MemStack::new(&mut mem_buf)
                );
            vm = (dt * aq.as_ref() - leja_x_sc_re[i-1]*qm.as_ref()) / scale
                + ((leja_x_sc_im[i-1]/scale).powi(2)) * vm.as_ref();
            pm += coeffs[i+1].re * vm.as_ref();

            // error est
            let err_est = (vm.norm_l2() * coeffs[i+1].re).abs();
            converged = err_est < self.tol * norm_u;
            iter += 2;
            if err_est > self.abort_tol {
                println!("Hit abort tol: {err_est:0.2e}. Consider a smaller time step size.");
                break;
            }
        }

        println!("CLaPM time (s): {}", clock.elapsed().as_secs_f64());
        (converged, iter)
    }

    /// Computes the linear combination: phi_0(dt*A)*v_0 + ... phi_k(dt*A)*v_k
    /// by leja polynomial approximation with optional substepping
    ///
    /// #Args
    /// * `ext_a_lo` - the linear operator A
    /// * `dt` - the stepsize
    /// * `vb` - a k-len sequence of rhs vectors corrosponding to each phi-function: phi_k
    pub fn leja_expmv_substep(&self, ext_a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64>
    {
        // setup the extended rhs vector
        let (ext_v, n) = ext_a_lo.get_v(vb);

        // allocate storage for result
        let mut expmv: Mat<f64> = faer::Mat::zeros(ext_v.nrows(), ext_v.ncols());

        // compute leja poly coeffs by divided difference
        let p = 16;  // dd taylor series poly order
        let coeffs: Col<c64> = dd_expm_taylor(
            &self.leja_x.slice(0, self.m), self.shift, self.scale, 1.0, p);

        // no substep
        let (_conv, _iters) = self.complex_conj_leja_expmv(
            expmv.as_mut(), ext_a_lo, dt, ext_v.as_ref(), self.shift, self.scale,
            coeffs.as_ref());
        println!("converged: {}, leja iters: {}, shift: {}, scale: {}",
            _conv, _iters, self.shift, self.scale);

        // extract the first n elements
        expmv.get(0..n, 0..1).to_owned()
    }

    /// Set the krylov reuse flag
    pub fn set_krylov_reuse(&mut self, krylov_reuse: bool) {
        self.krylov_reuse = krylov_reuse;
    }

    /// Set the max leja polynomial degree
    pub fn set_m(&mut self, m: usize) {
        self.m = m;
    }

    /// Set the shift and scale parameters and adjust leja sequence
    /// by splicing in additional points.
    ///
    /// #Args
    /// * `a` - is min real spectrum eig
    /// * `b` - is max real spectrum eig
    /// * `c` - is max imag spectrum eig magnitude
    /// * `splice_idx` - index where splice_lp are inserted
    /// * `splice_lp` - optional sequence of points to splice into the full sequence
    pub fn update_leja_splice(&mut self, a: f64, b: f64, c: f64, splice_idx: usize, splice_lp: LejaPoints) {
        let (leja_x, shift, scale) = self.leja_base.rescale(a, b, c);
        // construct the full leja sequence by splicing
        let first_lp = leja_x.slice(0, splice_idx);
        let last_lp = leja_x.slice(splice_idx, leja_x.n_leja());
        // splice into final sequence
        let leja_x_ext = first_lp.concat(vec![&splice_lp, &last_lp]);
        self.leja_x = leja_x_ext;
        self.shift = shift;
        self.scale = scale;
        self.n_leja_real = self.leja_x.n_leja_real();
    }

    /// Set the shift and scale parameters
    ///
    /// #Args
    /// * `a` - is min real spectrum eig
    /// * `b` - is max real spectrum eig
    /// * `c` - is max imag spectrum eig magnitude
    pub fn update_leja(&mut self, a: f64, b: f64, c: f64) {
        let (leja_x, shift, scale) = self.leja_base.rescale(a, b, c);
        self.leja_x = leja_x;
        self.shift = shift;
        self.scale = scale;
        self.n_leja_real = self.leja_x.n_leja_real();
    }

}

impl LinOpPhikvEvaluator for LejaPhiEval {
    fn apply_phi_k_v(&self, a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64> {
        let clock = std::time::Instant::now();
        // TODO: optionally auto-run apply_prepare here!
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
        let ext_a_lo = DynRefExtendedLinOp::new(dt, a_lo, &vbk);
        // TODO: optionally auto-run apply_prepare here!
        // compute phi_k(a_lo)*v
        self.leja_expmv_substep(&ext_a_lo, dt, &vbk)
    }

    fn apply_prepare(&mut self, a_lo: &dyn LinOp<f64>, dt: f64, v: MatRef<f64>) {
        const SPLICE_IDX: usize = 0;
        let clock = std::time::Instant::now();
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
            println!("=== Updating Spectrum Parameters ===");

            // building extended rhs if needed
            let mut a_lo_ext_flag = false;
            let mut v_ext = v.to_owned();
            let p = a_lo.nrows() - v.nrows();
            if p > 0 {
                a_lo_ext_flag = true;
                let mut e1 = Mat::zeros(p, 1);
                e1[(0, 0)] = 1.0;
                v_ext = faer::concat![[v.as_ref()], [e1]];
            }

            match self.spec_method.as_str() {
                "arnoldi" => {
                    let (a, b, c, ritz_re, ritz_im, q, h) = spectrum_arnoldi_iom(
                        a_lo, v_ext.as_ref(), dt, self.spec_iters, 4, false);
                    println!("Arnoldi Spectrum params: a={}, b={}, c={}", a, b, c);
                    // apply shift and scale to the ritz values
                    // splice complex conj ritz values into the leja sequence
                    if self.krylov_reuse == true {
                        // store the hessenberg matrix for re-use
                        self.arnld_q = Some(q);
                        self.arnld_h = Some(h);
                        self.ritz_re = Some(ritz_re.clone());
                        self.ritz_im = Some(ritz_im.clone());
                        let (lp_ritz, _, _) = LejaPoints::new(ritz_re, ritz_im)
                            .normalize(a, b, c)
                            // .reorder_conj_pairs()
                            .rescale(a, b, c);
                        self.update_leja_splice(a, b, c, SPLICE_IDX, lp_ritz);
                    } else {
                        self.update_leja(a, b, c);
                    }
                },
                "schur" => {
                    let (a, b, c, ritz_re, ritz_im, _eig_vecs) = spectrum_krylov_schur(
                        a_lo, v_ext.as_ref(), dt, self.spec_iters, 1.0e-6, false);
                    println!("Schur Spectrum params: a={}, b={}, c={}", a, b, c);
                    if self.krylov_reuse == true {
                        let (lp_ritz, _, _) = LejaPoints::new(ritz_re, ritz_im)
                            .normalize(a, b, c)
                            .mirror()
                            .rescale(a, b, c);
                        self.update_leja_splice(a, b, c, SPLICE_IDX, lp_ritz);
                    } else {
                        self.update_leja(a, b, c);
                    }
                },
                "power" => {
                    let (a, b, c, _b_k) = spectrum_pwr_itr(
                        a_lo, v_ext.as_ref(), dt, self.spec_iters, 1.0e-6);
                    self.update_leja(a, b, c);
                },
                "none" => {},
                _s => panic!("Unknown spec_method: {_s}. Pick one of: (arnoldi, schur, none)")
            }

            self.spec_norm = spec_norm;
        }
        println!("apply_prepare time (s): {}", clock.elapsed().as_secs_f64());
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
/// * `ext_a_lo` - linear operator, sparse mat or impls method to apply mat to vec
/// * `v0` - initial vector
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth
pub fn spectrum_arnoldi_iom(
    ext_a_lo: &dyn LinOp<f64>,
    v0: MatRef<f64>,
    scale: f64,
    n: usize,
    iom: usize,
    update_b: bool)
    -> (f64, f64, f64, Vec<f64>, Vec<f64>, Mat<f64>, Mat<f64>)
{
    // run arnoldi
    let (q, h, _bdwn) = arnoldi_lop(ext_a_lo, 1.0, v0, n, iom);

    // compute the ritz values
    let ritzv = h.eigenvalues().unwrap();

    // approx spetrum parameters
    let ritz_re = ritzv.iter().map(|v| scale * v.re()).collect::<Vec<f64>>();
    let ritz_im = ritzv.iter().map(|v| scale * v.im()).collect::<Vec<f64>>();
    let a = ritz_re.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let b = ritz_re.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let c = ritz_im.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    if update_b {
        return (*a, *b, *c, ritz_re, ritz_im, q, h)
    }
    // apply artificial spectrum bounds
    (a.min(-1.0e-2), b.max(0.0), *c, ritz_re, ritz_im, q, h)
}


#[cfg(test)]
mod test_matexp_leja {
    use assert_approx_eq::assert_approx_eq;
    use crate::matexp_krylov::KrylovExpm;
    use crate::mat_utils::mat_mat_approx_eq;
    use crate::matexp_pade::{matexp, phi};
    use crate::test_common::{gen_test_a, gen_test_b, gen_test_c};

    // bring everything from above (parent) module into scope
    use super::*;

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
        assert_approx_eq!(a, -1.0);
        assert_approx_eq!(b, -1.0e-3);
        assert_approx_eq!(c,  0.0);

        // build an extended linear operator
        let mut vbk: Vec<MatRef<f64>> = vec![];
        vbk.push(test_v.as_ref());
        let ext_a_lo = DynRefExtendedLinOp::new(1.0, &test_a, &vbk);
        let (ext_a, ext_b, ext_c, _, _, _, _) = spectrum_arnoldi_iom(&ext_a_lo, test_v.as_ref(), 1.0, 10, 10, true);

        // check for consistency
        assert_approx_eq!(a, ext_a);
        // assert_approx_eq!(b, ext_b);
        assert_approx_eq!(c, ext_c);

        // run power iteration
        let (pwr_a, _pwr_b, _pwr_c, _) = spectrum_pwr_itr(&ext_a_lo, test_v.as_ref(), 1.0, 40, 1e-5);
        // check for consistency
        assert_approx_eq!(a, pwr_a);

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
        assert_approx_eq!(a, min_b_re);
        // assert_approx_eq!(b, max_b_re);
        assert_approx_eq!(c, max_b_im);
    }

    #[test]
    fn test_taylor_expmv() {
        // test that exp(dt*A)*v products can be computed by a
        // taylor polynomial method.

        // Generate a test 3x3 matrix
        let (test_a, test_v) = gen_test_a();

        // compute the matrix matexp(dt*A)*v using dense impl
        let expm_tay = expm_taylor(test_a.as_ref(), 0.0, 1.0, 16);
        let expmv_tay_dense = expm_tay.as_ref() * test_v.as_ref();

        // compute the matrix matexp(dt*A)*v using matfree impl
        let lp = LejaPoints::new(vec![], vec![]);
        let leja_phikv_eval = LejaPhiEval::new(lp, 20, 0.0, 1.0, 1e-8, 1e-8, 20, "none", true);
        let mut expmv_tay_pm = faer::Mat::zeros(test_a.nrows(), 1);
        leja_phikv_eval.taylor_expmv(expmv_tay_pm.as_mut(),
            &test_a, 1.0, test_v.as_ref(), 0.0, 1.0);
        println!("{:?}", expmv_tay_dense.as_ref());
        println!("{:?}", expmv_tay_pm.as_ref());

        // Ensure results are consistent.
        mat_mat_approx_eq(
            expmv_tay_pm.as_ref(), expmv_tay_dense.as_ref(), 1e-8);
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
            // compute the spectrum parameters with arnoldi with incomplete orthogonalization
            let (a, b, c, _, _, _, _) = spectrum_arnoldi_iom(&test_m.as_ref(), test_v.as_ref(), 1.0, 10, 2, true);

            // apply shift and scaling to the leja sequence to match spectrum of the test_m linop
            let (lp_sc, shift, scale) = lp.rescale(a, b, c);
            println!("shift: {}, scale: {}", &shift, &scale);

            // compute the leja polynomial coeffs
            let coeffs = dd_expm_taylor(&lp_sc, shift, scale, 1.0, 16);

            // compute the matexp(dt*A)*v product via leja poly approx
            let leja_phikv_eval = LejaPhiEval::new(lp_sc, 80, shift, scale, 1e-8, 1e-8, 20, "arnoldi", true);
            let mut expmv_leja_pm: Mat<f64> = faer::Mat::zeros(test_m.nrows(), 1);
            let (conv, iter) = leja_phikv_eval.complex_conj_leja_expmv(expmv_leja_pm.as_mut(),
                &test_m, 1.0, test_v.as_ref(), shift, scale, coeffs.as_ref());
            println!("converged: {}, iter: {}", &conv, &iter);
            assert!(conv);
            assert!(iter > 0);

            // Ensure results are consistent with pade methods.
            let expmv_pade_dense = matexp(test_m.as_ref(), 1.0) * test_v.as_ref();
            println!("leja expmv: {:?}", &expmv_leja_pm);
            println!("pade expmv: {:?}", &expmv_pade_dense);
            mat_mat_approx_eq(
                expmv_leja_pm.as_ref(), expmv_pade_dense.as_ref(), 1e-8);
        }
    }

    #[test]
    fn test_leja_phikv() {
        // test that phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk can be computed by a
        // leja polynomial method.

        // load leja points
        let lp = LejaPoints::new_from_lib("leja_circle").slice(0, 100);

        // Generate a test matrix
        let (test_b, test_v) = gen_test_b();

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref(),];

        // setup the phi evaluator
        let mut leja_phikv_eval = LejaPhiEval::new(lp, 80, 0.0, 1.0, 1e-8, 1e-8, 20, "arnoldi", true);

        // compute the spectrum parameters with arnoldi with incomplete orthogonalization
        let (a, b, c, _, _, _, _) = spectrum_arnoldi_iom(&test_b.as_ref(), test_v.as_ref(), 1.0, 10, 2, true);
        // update the phi evaluator
        leja_phikv_eval.update_leja(a, b, c);

        // compute phi_0(dt*A)*b0
        // fn apply_phi_k_v(&self, a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64> {
        let dt = 1.0;
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
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

    fn _test_leja_ritz_phikv(dt: f64, test_b: Mat<f64>, test_v: Mat<f64>, krylov_reuse: bool, max_arnoldi_iters: usize) {
        // load leja points
        let lp = LejaPoints::new_from_lib("leja_circle").slice(0, 300);

        // generate vb vector: vb = [b0, b1, ... bk]
        let test_vb = vec![test_v.as_ref(),];

        // setup the phi evaluator
        let mut leja_phikv_eval = LejaPhiEval::new(
            lp, 280, 0.0, 1.0, 1e-15, 1e-10, max_arnoldi_iters,
            "arnoldi", krylov_reuse);

        // compute the spectrum parameters with arnoldi
        // and update the phi evaluator in one step
        let iom = 2;
        leja_phikv_eval.apply_prepare(&test_b, dt, test_v.as_ref());

        // print the first 10 leja+ritz points
        for i in 0..10 {
            let lp_re = leja_phikv_eval.leja_x.leja_re[i] * leja_phikv_eval.scale + leja_phikv_eval.shift;
            let lp_im = leja_phikv_eval.leja_x.leja_im[i] * leja_phikv_eval.scale;
            println!("lp: {} + {}i", lp_re, lp_im);
        }
        // print the ritz values
        let (_a, _b, _c, ritz_re, ritz_im, q, h) = spectrum_arnoldi_iom(
            &test_b.as_ref(), test_v.as_ref(), dt, max_arnoldi_iters, iom, false);
        println!("ritz re: {:?}", ritz_re);
        println!("ritz im: {:?}", ritz_im);
        // let phi0mv_krylov = test_v.norm_l2() * (q.as_ref() * matexp_pade::matexp(h.as_ref(), 1.0)).col(0).as_mat();
        // println!("krylov phi_0(dt*A)*b0: {:?}", phi0mv_krylov.as_ref());

        // compute phi_0(dt*A)*b0
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        let phi0mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, dt, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi0mv_pade_dense = matexp(test_b.as_ref(), 1.0) * test_v.as_ref();
        println!("leja_ritz phi0mv: {:?}", &phi0mv_leja_pm);
        println!("pade phi0mv: {:?}", &phi0mv_pade_dense);
        mat_mat_approx_eq(
            phi0mv_leja_pm.as_ref(), phi0mv_pade_dense.as_ref(), 1e-7);
        // mat_mat_approx_eq(
        //     phi0mv_leja_pm.as_ref(), phi0mv_krylov.as_ref(), 1e-7);

        // generate vb vector: vb = [b0, b1, ... bk]
        let zeros = faer::Mat::zeros(test_v.nrows(), test_v.ncols());
        let test_vb = vec![zeros.as_ref(), test_v.as_ref()];

        // compute phi_0(dt*A)*b0 + ... phi_k(dt*A)*bk
        let ext_b_lo = DynRefExtendedLinOp::new(dt, &test_b, &test_vb);
        leja_phikv_eval.apply_prepare(&ext_b_lo, dt, test_vb[0].as_ref());
        let phi1mv_leja_pm: Mat<f64> = leja_phikv_eval.apply_phi_k_v(&ext_b_lo, dt, &test_vb);

        // Ensure results are consistent with pade methods.
        let phi1mv_pade_dense = phi(test_b.as_ref(), 1) * test_v.as_ref();
        println!("leja_ritz phi1mv: {:?}", &phi1mv_leja_pm);
        println!("pade phi1mv: {:?}", &phi1mv_pade_dense);
        mat_mat_approx_eq(
            phi1mv_leja_pm.as_ref(), phi1mv_pade_dense.as_ref(), 1e-7);
    }

    #[test]
    fn test_leja_phikv_small() {
        let dt = 1.0;
        let (test_b, test_v) = gen_test_b();
        _test_leja_ritz_phikv(dt, test_b, test_v, false, 10);
    }

    #[test]
    fn test_leja_phikv_large() {
        // similar test on a larger system
        let dt = 1.0;
        let (test_b, test_v) = gen_test_c(80);
        _test_leja_ritz_phikv(dt, 1.0*test_b, test_v, false, 20);
    }

    #[test]
    fn test_leja_phikv_small_krylov_reuse() {
        let dt = 1.0;
        let (test_b, test_v) = gen_test_b();
        _test_leja_ritz_phikv(dt, test_b, test_v, true, 10);
    }

    #[test]
    fn test_leja_phikv_large_krylov_reuse() {
        // similar test on a larger system
        let dt = 1.0;
        let (test_b, test_v) = gen_test_c(80);
        _test_leja_ritz_phikv(dt, 1.0*test_b, test_v, true, 20);
    }
}
