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
use faer::matrix_free::{eigen};
use faer::matrix_free::LinOp;
use crate::ode_sys::ExtendedLinOp;
use crate::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use std::cmp::{min, max};


/// Rescale leja points on the real line to bound the interval [a, b].
pub fn scale_leja(leja_re: &[f64], a: f64, b: f64) -> vec<f64>
{
    let leja_re_min = leja_re.iter().min().unwrap();
    let leja_re_max = leja_re.iter().max().unwrap();
    let scale = (leja_re_max - leja_re_min) / (b - a);
    // let out = (leja_re - leja_re_min) * scale  + a;
    let leja_re_scaled = leja_re.iter().map(|x| (x - leja_re_min) * scale + a).collect();
    leja_re_scaled
}

/// Rescale conjugate complex leja points to bound the interval [a, b, -c, +c].
pub fn scale_leja_conj_complex(leja_re: &[f64], leja_im: &[f64], a: f64, b: f64, c: f64) -> (vec<f64>, vec<f64>)
{
    (scale_leja(leja_re, a, b), scale_leja(leja_im, -c, c))
}


/// Reads leja points from a file
/// real leja points in [-2, 2]
/// complex conj leja points are on the unit circle.
pub struct LejaPoints {
    leja_re: vec<f64>,
    leja_im: vec<f64>,
    leja_x: Col<c64>,
    Xi: Mat<c64>,
}

impl LejaPoints {
    pub fn new(leja_re: vec<f64>, leja_im: vec<f64>) -> Self {
        let n_leja = leja_re.len();
        let leja_x: Col<c64> = Col::from_fn(
            n_leja, |i: usize| {c64::new(leja_re[i], leja_im[i])});

        // Build the matrix Xi =
        // [[leja_00,       0,       0],
        // [[      1, leja_11,       0],
        // [[      0,       1, leja_22],
        // [[      ...                ]]
        let mut Xi: Mat<c64> = faer::Mat::zeros(n_leja, n_leja);
        for i in 0..n_leja {
            Xi[(i, i)] = c64::new(leja_x.leja_re[i], leja_x.leja_im[i]);
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
    (1..=num).product() as f64
}

/// Compute the dense matrix exponential using tayler series
pub fn expm_taylor(A: Mat<c64>, shift: f64, scale: f64, p: usize) -> Mat<c64>
{
    let mut M: Mat<c64> = scale * A.as_ref();
    let mut ts_expm: Mat<c64> = faer::Mat::identity(M.nrows(), M.ncols());
    for i in 0..p {
        ts_expm.copy_from( ts_expm.as_ref() + M / factorial(i+1) );
        M.copy_from(A.as_ref() * M.as_ref());
    }
    shift.exp() * ts_expm
}

/// Compute leja divided differences using taylor series method
pub fn dd_expm_taylor(leja_x: &LejaPoints, shift: f64, scale: f64, h: f64, p: usize) -> Col<c64>
{
    let n_leja = leja_x.leja_re.len();
    let Xi_shift: Mat<c64> = shift * faer::Mat::identity(n_leja, n_leja)
        + scale * leja_x.Xi.as_ref();

    // compute mean
    let mut mu_ = Col::zeros(1);
    col_mean(mu_, h * (shift + scale * leja_x.leja_x.as_ref()));
    let mu = mu_[0];

    // shift to zero mean
    let z = Xi_shift - faer::Mat::identity(n_leja, n_leja) * mu;

    // scaling factor (in powers of 2)
    let s_scale = z.norm_l1();
    let s = max((s_scale.log() / (2.0).log()).ceil(), 1);
    let hs = 1.0 / (2.0).powf(s);

    // compute expm(Z)
    let mut F = expm_taylor(hs*h*z, 0.0, 1.0, p);

    // squaring
    for _i in 0..s {
        F = F.as_ref() * F.as_ref();
    }

    // reshift and extract first col
    ((h * mu).exp() * F.col(0)).to_owned()
}

/// Used for phi function evaluation at the leja points
/// Evaluates linear combinations of phi-function-vector products
/// using either the real leja point method (RelPM) or
/// the conj. complex conj leja point method (CLaPM).
pub struct LejaPhiEval {
    leja_x: LejaPoints,
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
    pub fn new(leja_x: LejaPoints, shift: f64, scale: f64, tol: f64) -> Self
    {
        Self {
            leja_x: leja_x,
            n_leja: leja_x.leja_x.nrows(),
            n_leja_real: leja_x.n_leja_real(),
            tol: tol,
            abort_tol: 1e10,
            shift,
            scale,
        }
    }

    /// ReLPM method of
    /// Ref:  L. Bergamaschi.  M. Caliari. A. Martinez and M. Vianello.
    ///       Comparing Leja and Krylov Approximations of Large Scale
    ///       Matrix Exponentials. Intl. Conf on Computational Science. 2006.
    fn real_leja_expmv(&self, mut pm: MatMut<f64>, ext_a_lo: &ExtendedLinOp, dt: f64, u: MatRef<f64>, shift: f64, scale: f64, coeffs: ColRef<f64>, n: usize) -> (bool, usize)
    {
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let err_est: f64 = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);
        let leja_x_sc = shift + scale * self.leja_x.leja_re ;

        // first term of leja polynomial
        pm = coeffs[(0)] * u;
        let mut vm = u.to_owned();
        let mut err_est = 0.0;

        // compute leja poly and check for convergence each iter
        for i in 0..n {
            if converged {
                break;
            }
            vm = (dt * ext_a_lo.apply(vm) - leja_x_sc[i-1]*vm) / scale;
            // leja polynomial update
            pm = pm + coeffs[i] * vm;

            // check error estimate
            err_est = (coeffs[i]*pm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
        }
        (converged, iter)
    }

    /// Complex conjugate leja point method (CLaPM).
    fn complex_conj_leja_expmv(&self, mut pm: MatMut<f64>, ext_a_lo: &ExtendedLinOp, dt: f64, u: MatRef<f64>, shift: f64, scale: f64, coeffs_re: ColRef<f64>, coeffs_im: ColRef<f64>) -> (bool, usize)
    {
        let mut iter: usize = 1;
        let norm_u: f64 = u.norm_l2();
        let err_est = 2. * norm_u;
        let mut converged: bool = (err_est == 0.);
        let leja_x_sc_re = shift + scale * self.leja_x.leja_re;
        let leja_x_sc_im = scale * self.leja_x.leja_im;

        // use the real leja point method for all real case
        if self.n_leja_real >= self.n_leja {
            real_leja_expmv(pm, ext_a_lo, dt, u, shift, scale, coeffs_re)
        }

        // first term of leja polynomial
        pm = coeffs_re[(0)] * u;
        let mut vm = u.to_owned();
        let mut err_est = 0.0;

        // build leja polynomial for leading real points
        for i in 1..self.n_leja_real {
            if converged {
                break;
            }
            vm = (dt * ext_a_lo.apply(vm) - leja_x_sc_re[i-1]*vm) / scale;
            // leja polynomial update
            pm = pm + coeffs_re[i] * vm;

            // check error estimate
            err_est = (coeffs_re[i] * pm.norm_l2()).abs();
            converged = err_est < self.tol * norm_u;
            iter += 1;
        }

        // build remaining leja polynomial suported at
        // conjugate complex points.
        for i in (self.n_leja_real..self.n_leja).step_by(2) {
            let mut qm = (dt * ext_a_lo.apply(vm.as_ref()) - leja_x_sc_re[i-1]*vm.as_ref()) / scale;
            pm += coeffs_re[i] * qm;

            vm = (dt * ext_a_lo.apply(qm.as_ref()) - leja_x_sc_re[i-1]*qm.as_ref()) / scale
                + ((leja_x_sc_im[i-1]/scale).powi(2)) * vm.as_ref();
            pm += coeffs_re[i+1] * vm.as_ref();

            // error est
            let norm_vm = vm.as_ref().norm_l2();
            let err_est = (norm_vm * coeffs_re[i+1]).abs();
            converged = err_est < self.tol * norm_u;
            iter += 2;
        }
        (converged, iter)
    }

    pub fn leja_expmv_substep(&self, ext_a_lo: &ExtendedLinOp, dt: f64) -> Mat<f64>
    {
        // setup the extended rhs vector
        let (ext_v, n) = ext_a_lo.get_v(vb);

        // allocate storage for result
        let mut expmv: Mat<f64> = faer::Mat::zeros(ext_v.nrows(), ext_v.ncols());

        // compute leja poly coeffs by divided difference
        let coeffs: Mat<c64> = dd_expm_taylor(
            &self.leja_x, self.shift, self.scale, 1.0, 16);
        let coeffs_re = Col::from_fn(self.n_leja, |i: usize| {coeffs[i].re()});
        let coeffs_im = Col::from_fn(self.n_leja, |i: usize| {coeffs[i].im()});

        // no substep
        let (conv_, iters_) = complex_conj_leja_expmv(
            expmv.as_mut(), ext_a_lo, dt, ext_v.as_ref(), self.shift, self.scale,
            coeffs_re.as_ref(), coeffs_im.as_ref());

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
    pub fn update_shift_scale_gershgorin(&mut self, ext_a_lo: &ExtendedLinOp) {
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
