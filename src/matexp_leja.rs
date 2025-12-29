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
use crate::ode_sys::{ExtendedLinOp, DynRefExtendedLinOp};
use crate::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use std::cmp::{min, max};


/// Rescale leja points on the real line to bound the interval [a, b].
pub fn scale_leja(leja_re: &[f64], a: f64, b: f64) -> Vec<f64>
{
    let leja_re_min = leja_re.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let leja_re_max = leja_re.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let scale = (leja_re_max - leja_re_min) / (b - a);
    // let out = (leja_re - leja_re_min) * scale  + a;
    let leja_re_scaled = leja_re.iter().map(|x| (x - leja_re_min) * scale + a).collect();
    leja_re_scaled
}

/// Rescale conjugate complex leja points to bound the interval [a, b, -c, +c].
pub fn scale_leja_conj_complex(leja_re: &[f64], leja_im: &[f64], a: f64, b: f64, c: f64) -> (Vec<f64>, Vec<f64>)
{
    (scale_leja(leja_re, a, b), scale_leja(leja_im, -c, c))
}


/// Reads leja points from a file
/// real leja points in [-2, 2]
/// complex conj leja points are on the unit circle.
pub struct LejaPoints {
    leja_re: Col<f64>,
    leja_im: Col<f64>,
    leja_x: Col<c64>,
    Xi: Mat<c64>,
}

impl LejaPoints {
    pub fn new(leja_re_v: Vec<f64>, leja_im_v: Vec<f64>) -> Self {
        assert!(leja_re_v.len() == leja_im_v.len());
        let n_leja = leja_re_v.len();
        let leja_re: Col<f64> = faer::Col::from_fn(n_leja, |i| {leja_re_v[i]});
        let leja_im: Col<f64> = faer::Col::from_fn(n_leja, |i| {leja_im_v[i]});
        let leja_x: Col<c64> = Col::from_fn(
            n_leja, |i: usize| {c64::new(leja_re_v[i], leja_im_v[i])});

        // Build the matrix Xi =
        // [[leja_00,       0,       0],
        // [[      1, leja_11,       0],
        // [[      0,       1, leja_22],
        // [[      ...                ]]
        let mut Xi: Mat<c64> = faer::Mat::zeros(n_leja, n_leja);
        for i in 0..n_leja {
            Xi[(i, i)] = c64::new(leja_re[i], leja_im[i]);
            if i+1 < n_leja {
                Xi[(i+1, i)] = c64::new(1.0, 0.0);
            }
        }

        Self {
            leja_re,
            leja_im,
            leja_x,
            Xi,
        }
    }

    pub fn new_from_col(leja_re: Col<f64>, leja_im: Col<f64>) -> Self {
        assert!(leja_re.nrows() == leja_im.nrows());
        let n_leja = leja_re.nrows();
        let leja_x: Col<c64> = Col::from_fn(
            n_leja, |i: usize| {c64::new(leja_re[i], leja_im[i])});
        let mut Xi: Mat<c64> = faer::Mat::zeros(n_leja, n_leja);
        for i in 0..n_leja {
            Xi[(i, i)] = c64::new(leja_re[i], leja_im[i]);
            if i+1 < n_leja {
                Xi[(i+1, i)] = c64::new(1.0, 0.0);
            }
        }
        Self {
            leja_re,
            leja_im,
            leja_x,
            Xi,
        }
    }

    /// Create leja points from txt file of format
    ///     leja_re_0, leja_im_0
    ///     leja_re_1, leja_im_1
    ///     ...
    ///     leja_re_n, leja_im_n
    pub fn new_from_str(file_str: &str) {
        // parse file content string
    }

    /// Number of leading leja points on the real axis
    pub fn n_leja_real(&self) -> usize {
        // count number of leading nonzeros in vector
        let mut nz: usize = 0;
        for i in 0..self.leja_re.nrows() {
            if self.leja_re[i] != 0.0 && self.leja_im[i] == 0.0 {
                nz += 1;
                break;
            }
        }
        nz
    }

    /// Number of leja points
    pub fn n_leja(&self) -> usize {
        self.leja_re.nrows()
    }

    /// Shifted and scaled leja points
    pub fn leja_sc(&self, shift: f64, scale: f64) -> (Col<f64>, Col<f64>) {
        let leja_sc_re = faer::Col::full(self.n_leja(), shift) + scale * self.leja_re.as_ref();
        let leja_sc_im = scale * self.leja_im.as_ref();
        // split complex and real part
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
}

/// compute ellipse shift and scale parameters
pub fn ellipse_shift_scale(a: f64, b: f64, c: f64) -> (f64, f64) {
    // ellipse half axes
    let hax1 = (b-a)/2.;
    let hax2 = c;
    let shift = (a + b) / 2.;
    let scale = (hax1 + hax2) / 2.;
    (shift, scale)
}

/// computes the factorial
pub fn factorial(num: usize) -> f64 {
    (1..=num).product::<usize>() as f64
}

/// Compute the dense matrix exponential using tayler series
pub fn expm_taylor(A: Mat<c64>, shift: f64, scale: f64, p: usize) -> Mat<c64>
{
    let mut M: Mat<c64> = scale * A.as_ref();
    let mut ts_expm: Mat<c64> = faer::Mat::identity(M.nrows(), M.ncols());
    for i in 0..p {
        ts_expm.copy_from( ts_expm.as_ref() + M.as_ref() / factorial(i+1) );
        M.copy_from(A.as_ref() * M.as_ref());
    }
    shift.exp() * ts_expm
}

/// Compute leja divided differences using taylor series method
pub fn dd_expm_taylor(leja_x: &LejaPoints, shift: f64, scale: f64, h: f64, p: usize) -> Col<c64>
{
    let n_leja = leja_x.n_leja();
    let eye = faer::Mat::<c64>::identity(n_leja, n_leja);

    let Xi_shift: Mat<c64> = shift * eye.as_ref()
        + scale * leja_x.Xi.as_ref();

    // compute mean
    let mut mu_: Col<c64> = Col::zeros(1);
    // col_mean(mu_, h * (shift + scale * leja_x.leja_x.as_ref()));
    col_mean(mu_.as_mut(), (h * leja_x.Xi.as_ref()).as_ref(), NanHandling::Ignore);
    let mu = mu_[0];

    // shift to zero mean
    let z = Xi_shift - faer::Scale(mu) * eye.as_ref();

    // scaling factor (in powers of 2)
    let s_scale = z.norm_l1();
    let s = max((s_scale.ln() / (2.0 as f64).ln()).ceil() as i32, 1);
    let hs = 1.0 / (2.0 as f64).powi(s);

    // compute expm(Z)
    let mut F = expm_taylor(hs*h*z, 0.0, 1.0, p);

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
    leja_x: LejaPoints,
    n: usize,
    n_leja: usize,
    n_leja_real: usize,
    tol: f64,
    abort_tol: f64,
    shift: f64,
    scale: f64,
}


/// Taylor series evaluation of
/// linear combinations of phi-function-vector products
pub struct TaylorPhiEval {
}


/// Leja point phi function evaluator
impl LejaPhiEval {
    pub fn new(leja_x: LejaPoints, n: usize, a: f64, b: f64, c: f64, tol: f64) -> Self
    {
        let (shift, scale) = ellipse_shift_scale(a, b, c);
        Self {
            n: n,
            n_leja: (&leja_x).n_leja(),
            n_leja_real: (&leja_x).n_leja_real(),
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
    fn real_leja_expmv(
        &self,
        mut pm: MatMut<f64>,
        ext_a_lo: &DynRefExtendedLinOp,
        dt: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        coeffs: ColRef<f64>,
        n: usize) -> (bool, usize)
    {
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let err_est: f64 = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);

        // shift and scale leja points to align to the spectrum parameters
        let (leja_x_sc, _leja_x_sc_im) = self.leja_x.leja_sc(shift, scale);

        // first term of leja polynomial
        pm.copy_from(coeffs[0] * u);
        let mut vm = u.to_owned();
        let mut av = u.to_owned();
        let mut err_est = 0.0;

        // compute leja poly and check for convergence each iter
        for i in 1..n {
            if converged {
                break;
            }
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                faer::get_global_parallelism(),
                MemStack::new(&mut MemBuffer::new(StackReq::empty()))
                );
            vm = (dt * av.as_ref() - leja_x_sc[i-1]*vm) / scale;
            // leja polynomial update
            pm.copy_from( pm.as_ref() + coeffs[i] * vm.as_ref() );

            // check error estimate
            err_est = (coeffs[i]*pm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
            if err_est > self.abort_tol {
                break;
            }
        }
        (converged, iter)
    }

    /// Complex conjugate leja point method (CLaPM).
    fn complex_conj_leja_expmv(
        &self, mut pm: MatMut<f64>,
        ext_a_lo: &DynRefExtendedLinOp,
        dt: f64,
        u: MatRef<f64>,
        shift: f64,
        scale: f64,
        coeffs_re: ColRef<f64>,
        coeffs_im: ColRef<f64>,
        n: usize) -> (bool, usize)
    {
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let err_est = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);

        // shift and scale leja points to align to the spectrum parameters
        let (leja_x_sc_re, leja_x_sc_im) = self.leja_x.leja_sc(shift, scale);

        // use the real leja point method for all real case
        if self.n_leja_real >= n {
            return self.real_leja_expmv(
                pm, ext_a_lo, dt, u, shift, scale, coeffs_re, n)
        }

        // first term of leja polynomial
        pm.copy_from( coeffs_re[0] * u );
        let mut vm = u.to_owned();
        let mut av = u.to_owned();
        let mut aq = u.to_owned();
        let mut err_est = 0.0;

        // compute leja polynomial terms for leading real points
        for i in 1..self.n_leja_real {
            if converged {
                break;
            }
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                faer::get_global_parallelism(),
                MemStack::new(&mut MemBuffer::new(StackReq::empty()))
                );
            vm = (dt * av.as_ref() - leja_x_sc_re[i-1]*vm) / scale;
            // leja polynomial update
            pm.copy_from( pm.as_ref() + coeffs_re[i] * vm.as_ref() );

            // check error estimate
            err_est = (coeffs_re[i]*pm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
        }

        // compute remaining leja polynomial terms suported at
        // conjugate complex points.
        for i in (self.n_leja_real..self.n_leja).step_by(2) {
            ext_a_lo.apply(av.as_mut(), vm.as_ref(),
                faer::get_global_parallelism(),
                MemStack::new(&mut MemBuffer::new(StackReq::empty()))
                );
            let qm = (dt * av.as_ref() - leja_x_sc_re[i-1]*vm.as_ref()) / scale;
            pm += coeffs_re[i] * qm.as_ref();

            ext_a_lo.apply(aq.as_mut(), qm.as_ref(),
                faer::get_global_parallelism(),
                MemStack::new(&mut MemBuffer::new(StackReq::empty()))
                );
            vm = (dt * aq.as_ref() - leja_x_sc_re[i-1]*qm.as_ref()) / scale
                + ((leja_x_sc_im[i-1]/scale).powi(2)) * vm.as_ref();
            pm += coeffs_re[i+1] * vm.as_ref();

            // error est
            let norm_vm = vm.as_ref().norm_l2();
            let err_est = (norm_vm * coeffs_re[i+1]).abs();
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
        let coeffs_re = Col::from_fn(self.n_leja, |i: usize| {coeffs[i].re()});
        let coeffs_im = Col::from_fn(self.n_leja, |i: usize| {coeffs[i].im()});

        // no substep
        let (_conv, _iters) = self.complex_conj_leja_expmv(
            expmv.as_mut(), ext_a_lo, dt, ext_v.as_ref(), self.shift, self.scale,
            coeffs_re.as_ref(), coeffs_im.as_ref(), self.n);

        // extract the first n elements
        expmv.get(0..n, 0..1).to_owned()
    }

    /// set the shift and scale parameters
    pub fn set_shift_scale(&mut self, shift: f64, scale: f64) {
        self.shift = shift;
        self.scale = scale;
    }

    /// set the shift and scale parameters
    /// a is min magnitude real spectrum eig (typically 0)
    /// b is max magnitude real spectrum eig (possibly negative number)
    /// c is max imag spectrum eig magnitude
    pub fn set_shift_scale_abc(&mut self, a: f64, b: f64, c: f64) {
        (self.shift, self.scale) = ellipse_shift_scale(a, b, c);
    }

    /// auto update the shift and scale parameters
    /// using the Gershgorin circle theorem
    /// diagonals of matrix are centers of disks
    /// sum of each row is radius of each disk
    /// take max radius and max diag + radius as spectrum bounds
    pub fn update_shift_scale_gershgorin(&mut self, ext_a_lo: &DynRefExtendedLinOp) {
        //
        let a: f64 = 0.0;
        let mut b: f64 = 0.0;
        let mut c: f64 = 0.0;

        //let diag = ext_a_lo.inner_lop.apply(eye);
        //let row_sums = ext_a_lo.inner_lop.apply(ones);

        // ellipse half axes
        let hax1 = (b-a)/2.;
        let hax2 = c;
        self.shift = (a + b) / 2.0;
        self.scale = (hax1 + hax2) / 2.;
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


#[cfg(test)]
mod test_matexp_krylov {
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

}
