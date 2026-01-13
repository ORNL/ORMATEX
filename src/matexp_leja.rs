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
use faer::stats::{col_mean, NanHandling};
use faer::matrix_free::LinOp;
use faer::complex::ComplexFloat;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};

use std::cmp::{max};
use statrs::function::{factorial};
use csv;

use crate::ode_sys::{DynRefExtendedLinOp};
use crate::matexp_traits::{LinOpPhikvEvaluator};
use crate::arnoldi::{arnoldi_lop};

/// Pre-generated Leja points from file
/// Real leja points in [-2, 2]
/// TODO: Generate leja points in [-1, 1]
const LEJA_REAL_CSV: &str = std::include_str!("leja_points_real");
/// Complex conjugate leja points are on the unit circle.
const LEJA_CIRCLE_CSV: &str = std::include_str!("leja_points_circle");


/// compute ellipse shift and scale parameters
pub fn ellipse_shift_scale(a: f64, b: f64, c: f64) -> (f64, f64) {
    // ellipse half axes
    let hax1 = (b-a)/2.;
    let hax2 = c;
    let shift = (a + b) / 2.;
    let scale = (hax1 + hax2) / 2.;
    (shift, scale)
}

/// Rescale the leja points to bound the interval [a, b, -c, +c].
///
/// Returns:
///         (leja_re, leja_im, n_real, scale, shift)
pub fn shift_scale_leja(leja_re: ColRef<f64>, leja_im: ColRef<f64>, a: f64, b: f64, c: f64)
    -> (Col<f64>, Col<f64>, f64, f64)
{
    assert!(leja_re.nrows() == leja_im.nrows());
    let (shift, scale) = ellipse_shift_scale(a, b, c);
    let (hax1, hax2) = ( (b - a) / 2.0, c );
    // normalize half axes to capacity 1
    let (h1, h2) = (hax1 / scale, hax2 / scale);
    // shift and scale the leja points
    let leja_re_scaled = h1 * leja_re.as_ref();
    let leja_im_scaled = h2 * leja_im.as_ref();
    (leja_re_scaled, leja_im_scaled, shift, scale)
}


/// The Leja points
#[derive(Debug)]
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

    /// Create leja points from txt file of format
    ///     leja_re_0, leja_im_0
    ///     leja_re_1, leja_im_1
    ///     ...
    ///     leja_re_n, leja_im_n
    pub fn new_from_file(file_str: &str) -> Self {
        // parse file content string
        todo!("Implement leja points from user file.");
    }

    /// Leja points from pre-generated library
    pub fn new_from_lib(lib_str: &str) -> Self {
        let lp_str = match lib_str {
            "leja_real" => LEJA_REAL_CSV,
            "leja_circle" => LEJA_CIRCLE_CSV,
            _ => panic!("Invalid lib_str.")
            };
        // storage for real and complex leja points
        let mut real_lp: Vec<f64> = vec![];
        let mut complex_lp: Vec<f64> = vec![];
        // parse the leja point string
        let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_reader(lp_str.as_bytes());
        for result in rdr.records() {
            let record = result.expect("parsing record failed");
            let re: f64 = record.get(0).unwrap().replace(" ", "").parse::<f64>().unwrap();
            let im: f64 = record.get(1).unwrap().replace(" ", "").parse::<f64>().unwrap();
            real_lp.push(re);
            complex_lp.push(im);
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
        let tol = 1.0e-25;
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

    /// Prepend additional points to the sequence.
    /// Useful to build a hybrid leja-hermite sequence.
    pub fn join(&self, other: &LejaPoints) -> Self {
        // join
        let full_re = faer::concat![[other.leja_re.as_mat()], [self.leja_re.as_mat()]];
        let full_im = faer::concat![[other.leja_im.as_mat()], [self.leja_im.as_mat()]];
        // re-init
        Self::new_from_col(full_re.col(0).to_owned(), full_im.col(0).to_owned())
    }

    /// The first n leja points from the sequence
    pub fn head(&self, n: usize) -> Self {
        assert!(n < self.n_leja());
        Self::new_from_col(self.leja_re.get(0..n).to_owned(), self.leja_im.get(0..n).to_owned())
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
    let s = max((s_scale.ln() / (2.0 as f64).ln()).ceil() as i32, 1);
    let hs = 1.0 / (2.0 as f64).powi(s);

    // compute expm(Z)
    let mut F = expm_taylor((hs*h*z).as_ref(), 0.0, 1.0, p);

    // squaring
    for _i in 0..s {
        F = F.as_ref() * F.as_ref();
    }

    // reshift and extract first col
    faer::Scale( (h * mu).exp() ) * F.col(0)
}

/// Used for phi function evaluation at the leja points
/// Evaluates linear combinations of phi-function-vector products
/// using either the real leja point method (RelPM) or
/// the conj. complex conj leja point method (CLaPM).
pub struct LejaPhiEval {
    /// the leja points
    leja_x: LejaPoints,
    /// maximum leja polynomial degree
    m: usize,
    n_leja: usize,
    n_leja_real: usize,
    n_leja_zero: usize,
    tol: f64,
    abort_tol: f64,
    shift: f64,
    scale: f64,
}


/// Leja point phi function evaluator
impl LejaPhiEval {
    pub fn new(leja_x: LejaPoints, m: usize, shift: f64, scale: f64, tol: f64) -> Self
    {
        Self {
            m: m,
            n_leja: (&leja_x).n_leja(),
            n_leja_real: (&leja_x).n_leja_real(),
            n_leja_zero: (&leja_x).n_leja_zero(),
            leja_x: leja_x,
            tol: tol,
            abort_tol: 1e10,
            shift: shift,
            scale: scale,
        }
    }

    /// ReLPM method of
    /// Ref:  L. Bergamaschi.  M. Caliari. A. Martinez and M. Vianello.
    ///       Comparing Leja and Krylov Approximations of Large Scale
    ///       Matrix Exponentials. Intl. Conf on Computational Science. 2006.
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
        println!("=== Running ReLPM ===");
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let err_est: f64 = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);

        // shift and scale leja points to align to the spectrum parameters
        let (leja_x_sc, _leja_x_sc_im) = self.leja_x.leja_sc(shift, scale);

        // first term of leja polynomial
        pm.copy_from(coeffs[0].re * u);
        let mut vm = u.to_owned();
        let mut av = u.to_owned();
        let mut err_est = 0.0;

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
                break;
            }
        }
        (converged, iter)
    }

    /// Use simple taylor series method to estimate the action of
    /// the matrix exponential on a vector.
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
        println!("=== Running TS ===");
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let mut err_est = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);

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
        (converged, iter)
    }

    /// Complex conjugate leja point method (CLaPM).
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
        println!("=== Running CLaPM ===");
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let mut err_est = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);

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

        // first term of leja polynomial
        pm.copy_from( coeffs[0].re * u );
        let mut vm = u.to_owned();
        let mut av = u.to_owned();
        let mut aq = u.to_owned();

        let par = faer::get_global_parallelism();
        let mut mem_buf = MemBuffer::new(ext_a_lo.apply_scratch(u.ncols(), par));

        // compute leja polynomial terms for leading real points
        for i in 1..=self.n_leja_real {
            if converged {
                break;
            }
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
        for i in (self.n_leja_real+1..self.m).step_by(2) {
            if converged {
                break;
            }
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
                break;
            }
        }
        (converged, iter)
    }

    pub fn leja_expmv_substep(&self, ext_a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64>
    {
        // setup the extended rhs vector
        let (ext_v, n) = ext_a_lo.get_v(vb);

        // allocate storage for result
        let mut expmv: Mat<f64> = faer::Mat::zeros(ext_v.nrows(), ext_v.ncols());

        // compute leja poly coeffs by divided difference
        let p = 16;  // dd taylor series poly order
        let coeffs: Col<c64> = dd_expm_taylor(
            &self.leja_x, self.shift, self.scale, 1.0, p);

        // no substep
        let (_conv, _iters) = self.complex_conj_leja_expmv(
            expmv.as_mut(), ext_a_lo, dt, ext_v.as_ref(), self.shift, self.scale,
            coeffs.as_ref());

        // extract the first n elements
        expmv.get(0..n, 0..1).to_owned()
    }

    /// Set the max leja polynomial degree
    pub fn set_m(&mut self, m: usize) {
        self.m = m
    }

    /// set the shift and scale parameters
    /// a is min magnitude real spectrum eig (typically 0)
    /// b is max magnitude real spectrum eig (possibly negative number)
    /// c is max imag spectrum eig magnitude
    pub fn update_leja(&mut self, a: f64, b: f64, c: f64) {
        let (new_leja_x, shift, scale) = self.leja_x.rescale(a, b, c);
        (self.shift, self.scale) = (shift, scale);
        self.leja_x = new_leja_x;
    }

}

impl <'a> LinOpPhikvEvaluator <'a> for LejaPhiEval {
    fn apply_phi_k_v(&self, a_lo: &DynRefExtendedLinOp, dt: f64, vb: &Vec<MatRef<f64>>) -> Mat<f64> {
        self.leja_expmv_substep(a_lo, dt, vb)
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
        // compute phi_k(a_lo)*v
        self.leja_expmv_substep(&ext_a_lo, dt, &vbk)
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
    (a, b, c)
}

/// Using power iteration to estimate
/// spectrum paramters.
pub fn spectrum_pwr_itr(ext_a_lo: &dyn LinOp<f64>, v0: MatRef<f64>, n: usize, tol: f64) -> (f64, f64, f64) {
    // run power iter
    let mut b_k = v0.to_owned();
    let mut b_k1 = v0.to_owned();
    let mut eig_old = 1.0e20;
    let mut eig_new = 0.0;
    for i in 0..n {
        ext_a_lo.apply(
            b_k1.as_mut(),
            b_k.as_ref(),
            faer::get_global_parallelism(),
            MemStack::new(&mut MemBuffer::new(StackReq::empty()))
        );
        eig_new = (b_k.transpose() * b_k1.as_ref())[(0,0)] / (b_k.transpose() * b_k.as_ref())[(0,0)];
        let b_k1_norm = b_k1.norm_l2();
        // b_k.copy_from(b_k1.as_ref() / b_k1_norm);
        b_k = b_k1.as_ref() / b_k1_norm;
        let eig_diff = eig_new - eig_old;
        if eig_diff.abs() < tol {
            break;
        }
        eig_old = eig_new;
    }

    // estimate spetrum parameters from dominant eigv
    let a = -eig_new.abs();
    let b = 0.0;
    let c = 0.0;
    (a, b, c)
}

/// Using arnoldi iteration to estimate
/// spectrum paramters.
///
/// #Args
/// * `ext_a_lo` - linear operator, sparse mat or impls method to apply mat to vec
/// * `v0` - initial vector
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth
pub fn spectrum_arnoldi_iom(ext_a_lo: &dyn LinOp<f64>, v0: MatRef<f64>, n: usize, iom: usize, update_b: bool) -> (f64, f64, f64) {
    // run arnoldi
    let (_q, h, _bdwn) = arnoldi_lop(ext_a_lo, 1.0, v0, n, iom);

    // compute the ritz values
    let ritzv = h.eigenvalues().unwrap();

    // approx spetrum parameters
    let ritz_re = ritzv.iter().map(|v| v.re()).collect::<Vec<f64>>();
    let ritz_im = ritzv.iter().map(|v| v.im()).collect::<Vec<f64>>();
    let a = ritz_re.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let b = ritz_re.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let c = ritz_im.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    if update_b {
        return (*a, *b, *c)
    }
    let b = 0.0;
    (*a, b, *c)
}


#[cfg(test)]
mod test_matexp_leja {
    use assert_approx_eq::assert_approx_eq;
    use crate::matexp_krylov::KrylovExpm;
    use crate::mat_utils::mat_mat_approx_eq;
    use crate::matexp_pade::{matexp, phi};

    // bring everything from above (parent) module into scope
    use super::*;

    fn gen_test_a() -> (Mat<f64>, Mat<f64>)
    {
        // Generate a test 3x3 matrix with pure real eigs
        let test_m = faer::mat![
            [-1.0e-1,  0.0,    0.0],
            [ 1.0e-1, -1.0,  0.0],
            [    0.0,  1.0, -1.0e-3],
            ];
        // Generate a test vector
        let test_v = faer::mat![
            [0.1],
            [0.2],
            [0.01],
            ];
        (test_m, test_v)
    }

    fn gen_test_b() -> (Mat<f64>, Mat<f64>)
    {
        // Generate a test 3x3 matrix with one real eig and
        // conjugate complex eigen pair
        let lambda_c = 1.0;
        let lambda_a = 0.5;
        let lambda_b = 0.1;
        let vs = 1.0;
        let test_m = faer::mat![
            [-lambda_a,    -vs,            0.0],
            [ lambda_a+vs, -lambda_b,      0.0],
            [    0.0,       lambda_b, -lambda_c],
            ];
        // eigs = [-1. +0.j       , -0.3+1.2083046j, -0.3-1.2083046j]:
        // Generate a test vector
        let test_v = faer::mat![
            [0.1],
            [0.2],
            [0.01],
            ];
        (test_m, test_v)
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
        let (a, b, c) = spectrum_arnoldi_iom(&test_a.as_ref(), test_v.as_ref(), 10, 2, true);
        println!("Spectrum params: a= {a}, b= {b}, c= {c}");
        assert_approx_eq!(a, -1.0);
        assert_approx_eq!(b, -1.0e-3);
        assert_approx_eq!(c,  0.0);

        // build an extended linear operator
        let mut vbk: Vec<MatRef<f64>> = vec![];
        vbk.push(test_v.as_ref());
        let ext_a_lo = DynRefExtendedLinOp::new(1.0, &test_a, &vbk);
        let (ext_a, ext_b, ext_c) = spectrum_arnoldi_iom(&ext_a_lo, test_v.as_ref(), 10, 10, true);

        // check for consistency
        assert_approx_eq!(a, ext_a);
        assert_approx_eq!(b, ext_b);
        assert_approx_eq!(c, ext_c);

        // run power iteration
        let (pwr_a, _pwr_b, _pwr_c) = spectrum_pwr_itr(&ext_a_lo, test_v.as_ref(), 40, 1e-5);
        // check for consistency
        assert_approx_eq!(a, pwr_a);

        // check spectrum parameters of matrix with conj complex eig pair
        let (a, b, c) = spectrum_arnoldi_iom(&test_b.as_ref(), test_v.as_ref(), 10, 2, true);
        println!("Spectrum params: a= {a}, b= {b}, c= {c}");
        assert_approx_eq!(a, -1.0);
        assert_approx_eq!(b, -0.3);
        assert_approx_eq!(c,  1.2083046);
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
        let leja_phikv_eval = LejaPhiEval::new(lp, 20, 0.0, 1.0, 1e-8);
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
        let lp_re = LejaPoints::new_from_lib("leja_real").head(40);
        let lp_clp = LejaPoints::new_from_lib("leja_circle").head(40);
        // println!("{:?}", &lp_re);
        // println!("{:?}", &lp_clp);

        // Generate a test 3x3 matricies
        let (test_a, test_v) = gen_test_a();
        let (test_b, _) = gen_test_b();
        let test_mats = vec![test_a, test_b];

        for test_m in test_mats.iter() {
            // compute the spectrum parameters with arnoldi with incomplete orthogonalization
            let (a, b, c) = spectrum_arnoldi_iom(&test_m.as_ref(), test_v.as_ref(), 10, 2, true);

            // apply shift and scaling to the leja sequence to match spectrum of the test_m linop
            let (lp_re_sc, _shift, _scale) = lp_re.rescale(a, b, c);
            let (lp_clp_sc, shift, scale) = lp_clp.rescale(a, b, c);

            // println!("{:?}", &lp_re_sc);
            // println!("{:?}", &lp_clp_sc);
            println!("shift: {}, scale: {}", &shift, &scale);

            // compute the leja polynomial coeffs
            let coeffs = dd_expm_taylor(&lp_clp_sc, shift, scale, 1.0, 16);

            // compute the matexp(dt*A)*v product via leja poly approx
            let leja_phikv_eval = LejaPhiEval::new(lp_clp_sc, 20, shift, scale, 1e-8);
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
        // Ensure results are consistent with pade methods.
    }

}
