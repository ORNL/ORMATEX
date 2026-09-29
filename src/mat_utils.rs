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
//! Dense and sparse matrix helpers and linear operators.
//!
//! This module collects small utilities used throughout the crate and its
//! tests: random matrix generators, approximate matrix comparison, real and
//! complex conversions, matrix powers, sparse conversions, and a factorial.
//! It also defines a crate-local [`LinOp`] trait, whose method
//! `apply_linop_to_vec` applies a (possibly state dependent) linear operator
//! to a vector, together with implementations for a finite difference
//! Jacobian-vector product ([`JacobianRhsLinOp`]), a sparse matrix
//! ([`JacobianMatLinOp`]) and the enum [`MatrixLinOp`] combining them.
use faer::prelude::*;
use faer_traits::ComplexField;
use faer_traits::RealField;
use num_traits::Float;
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use std::cell::RefCell;

/// Creates a matrix filled with standard normal samples.
///
/// Entries are independent samples of $N(0, 1)$ drawn from the thread local
/// random number generator.
///
/// # Arguments
///
/// * `n_rows` - number of rows
/// * `n_cols` - number of columns
///
/// # Returns
///
/// An `n_rows` by `n_cols` random matrix.
pub fn random_mat_normal<T>(n_rows: usize, n_cols: usize) -> Mat<T>
where
    T: RealField + Float,
{
    let omega: Mat<T> = Mat::from_fn(n_rows, n_cols, |_i, _j| {
        T::from::<f64>(thread_rng().sample(StandardNormal)).unwrap()
    });
    omega
}

/// Creates a matrix filled with uniform random samples.
///
/// Entries are independent samples drawn uniformly from the half-open
/// interval $[lb, ub)$ using the thread local random number generator.
///
/// # Arguments
///
/// * `n_rows` - number of rows
/// * `n_cols` - number of columns
/// * `lb` - lower bound of the sampling interval
/// * `ub` - upper bound of the sampling interval
///
/// # Returns
///
/// An `n_rows` by `n_cols` random matrix.
///
/// # Panics
///
/// Panics if `lb >= ub`.
pub fn random_mat_uniform<T>(n_rows: usize, n_cols: usize, lb: f64, ub: f64) -> Mat<T>
where
    T: RealField + Float,
{
    let uni_dist = Uniform::new(lb, ub);
    let omega: Mat<T> = Mat::from_fn(n_rows, n_cols, |_i, _j| {
        T::from::<f64>(thread_rng().sample(uni_dist)).unwrap()
    });
    omega
}

/// Asserts that two matrices are approximately equal.
///
/// Checks that the shapes match and that every pair of entries agrees to
/// within `tol`, using `assert_approx_eq!` (absolute difference).
///
/// # Arguments
///
/// * `a` - first matrix
/// * `b` - second matrix
/// * `tol` - absolute tolerance for each entry
///
/// # Panics
///
/// Panics if the shapes of `a` and `b` differ or if any entries differ by
/// `tol` or more.
pub fn mat_mat_approx_eq<T>(a: MatRef<T>, b: MatRef<T>, tol: T)
where
    T: RealField + Float,
{
    use assert_approx_eq::assert_approx_eq;
    assert_eq!(a.ncols(), b.ncols());
    assert_eq!(a.nrows(), b.nrows());
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_approx_eq!(a[(i, j)], b[(i, j)], tol);
        }
    }
}

/// Extracts the real part of a complex matrix.
///
/// # Arguments
///
/// * `a` - complex matrix
///
/// # Returns
///
/// A real matrix of the same size holding the real part of each entry of `a`.
pub fn real_mat<T: RealField + Float>(a: MatRef<num_complex::Complex<T>>) -> Mat<T> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)].re)
}

/// Converts a real matrix to a complex matrix and scales it by `dt`.
///
/// # Arguments
///
/// * `a` - real matrix
/// * `dt` - real scale factor, e.g. a time step
///
/// # Returns
///
/// The complex matrix $dt \cdot A$ with zero imaginary part, same size as `a`.
pub fn complex_mat_scale<T: RealField + Float>(a: MatRef<T>, dt: f64) -> Mat<num_complex::Complex<T>> {
    let dt = T::from(dt).unwrap();
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
        num_complex::Complex::new(a[(i, j)] * dt, T::from(0.0).unwrap())
    })
}

/// Computes the integer power $A^p$ of a square matrix.
///
/// Uses `p` successive matrix multiplications, so it costs $O(p n^3)$ for an
/// $n \times n$ matrix. For `p = 0` the identity matrix is returned.
///
/// # Arguments
///
/// * `a` - square matrix $A$
/// * `p` - non-negative integer power
///
/// # Returns
///
/// The matrix $A^p$, same size as `a`.
pub fn mat_pow<T>(a: MatRef<T>, p: usize) -> Mat<T>
where
    T: ComplexField,
{
    let mut ap_out: Mat<T> = Mat::identity(a.nrows(), a.ncols());
    for _i in 0..p {
        ap_out = a.as_ref() * ap_out.as_ref();
    }
    ap_out
}

/// Converts a dense matrix to a sparse column-major matrix.
///
/// Entries that are exactly zero are dropped. For testing ONLY.
///
/// # Arguments
///
/// * `a` - dense matrix
///
/// # Returns
///
/// A sparse matrix of the same size holding the nonzero entries of `a`.
pub fn dense_to_sprs<T>(a: MatRef<T>) -> SparseColMat<usize, T>
where
    T: RealField + Float,
{
    // create triplets
    let mut a_triplets = Vec::new();
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if a[(i, j)].abs() != T::from(0.0).unwrap() {
                a_triplets.push(faer::sparse::Triplet::new(i, j, a[(i, j)]));
            }
        }
    }
    let out =
        SparseColMat::<usize, T>::try_new_from_triplets(a.nrows(), a.ncols(), &a_triplets).unwrap();
    out
}

/// Computes the factorial $n!$ as an `f64`.
///
/// The product is accumulated in `usize` before conversion, so it overflows
/// for `num > 20` on 64-bit targets. `ufactorial(0)` is 1.
///
/// # Arguments
///
/// * `num` - the integer $n$
///
/// # Returns
///
/// The value $n!$.
pub fn ufactorial(num: usize) -> f64 {
    (1..=num).product::<usize>() as f64
}

/// Linear operator that may depend on the time and the state.
///
/// This is a crate-local trait, distinct from `faer::matrix_free::LinOp`.
pub trait LinOp<T>
where
    T: RealField + Float,
{
    /// Applies the linear operator to a vector.
    ///
    /// For an operator $A(t, x)$ this computes $s \thinspace A(t, x) \thinspace w$.
    ///
    /// # Arguments
    ///
    /// * `t` - time at which the operator is evaluated
    /// * `x` - state at which the operator is linearized (ignored by state
    ///   independent operators)
    /// * `w` - the vector to which the operator is applied
    /// * `s` - optional scale factor applied to the product; defaults to 1
    ///
    /// # Returns
    ///
    /// The product $s \thinspace A(t, x) \thinspace w$.
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T>;
}

/// Finite difference Jacobian-vector product operator.
///
/// If $A$ is the Jacobian of the right hand side $F$ (`frhs`), the
/// Jacobian-vector product is approximated by the forward difference
///
/// $$ A w \approx \frac{F(x + \varepsilon w) - F(x)}{\varepsilon} $$
///
/// with $\varepsilon = 0.5 \times 10^{-8} \Vert x\Vert_1$. The evaluation
/// $F(x)$ is cached and reused when the $\ell_1$ norm of `x` is unchanged
/// from the previous call.
#[derive(Clone)]
pub struct JacobianRhsLinOp<'a, T>
where
    T: RealField + Float,
{
    /// Function ref to RHS of the system
    frhs: &'a dyn Fn(T, MatRef<T>) -> Mat<T>,

    /// Pointer to storage for x vector cache
    x_tmp: RefCell<Mat<T>>,

    /// Pointer to storage for F(x) vector (RHS eval) cache
    fx_tmp: RefCell<Mat<T>>,
}

impl<'a, T> LinOp<T> for JacobianRhsLinOp<'a, T>
where
    T: RealField + Float,
{
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T> {
        let x_norm_l1 = x.norm_l1();
        if x_norm_l1 == self.x_tmp.borrow().as_ref().norm_l1() {
            // we can reuse prior frhs eval
        } else {
            // must re-eval frhs (expensive)
            *self.x_tmp.borrow_mut() = x.to_owned();
            *self.fx_tmp.borrow_mut() = (self.frhs)(t, x);
        }
        // If A is a Jacobian, a Jacobian-vector product can be
        // given as $`J w \approx (F(x + \eps w) - F(x)) / \eps `$
        // let mut jw: Mat<T> = a * w_col;
        let eps = T::from(0.5e-8).unwrap() * x_norm_l1;
        let ieps = T::from(1.0).unwrap() / eps;
        let x_pert = x + faer::Scale(eps) * w.as_ref();
        let scaler = s.unwrap_or(T::from(1.0).unwrap());
        let Jw: Mat<T> = faer::Scale(scaler)
            * ((self.frhs)(t, x_pert.as_ref()) - (self.fx_tmp.borrow().as_ref()))
            * faer::Scale(ieps);
        Jw
    }
}
impl<'a, T> JacobianRhsLinOp<'a, T>
where
    T: RealField + Float,
{
    /// Creates a finite difference Jacobian operator from a right hand side function.
    ///
    /// # Arguments
    ///
    /// * `frhs` - the right hand side function $F(t, x)$
    /// * `dim` - dimension used to allocate the internal caches
    pub fn new(frhs: &'a dyn Fn(T, MatRef<T>) -> Mat<T>, dim: usize) -> Self {
        Self {
            frhs,
            x_tmp: RefCell::new(faer::Mat::zeros(dim, dim)),
            fx_tmp: RefCell::new(faer::Mat::zeros(dim, dim)),
        }
    }
}

/// Wrapper around a sparse matrix reference to apply it to a vector.
///
/// The operator is state and time independent.
pub struct JacobianMatLinOp<'a, T>
where
    T: RealField + Float,
{
    a_mat: SparseColMatRef<'a, usize, T>,
}
impl<'a, T> JacobianMatLinOp<'a, T>
where
    T: RealField + Float,
{
    /// Creates an operator wrapping a sparse matrix.
    ///
    /// # Arguments
    ///
    /// * `a_mat` - reference to the sparse matrix $A$
    pub fn new(a_mat: SparseColMatRef<'a, usize, T>) -> Self {
        Self { a_mat }
    }
}
impl<'a, T> LinOp<T> for JacobianMatLinOp<'a, T>
where
    T: RealField + Float,
{
    fn apply_linop_to_vec(&self, _t: T, _x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T> {
        self.a_mat * w * faer::Scale(s.unwrap_or(T::from(1.0).unwrap()))
    }
}

/// Enum of linear operators.
#[derive(Clone)]
pub enum MatrixLinOp<'a, T>
where
    T: RealField + Float,
{
    /// A general linear operator object, applied through its `apply_linop_to_vec`
    Lop(&'a dyn LinOp<T>),
    /// A constant sparse matrix
    MatLop(SparseColMatRef<'a, usize, T>),
    /// A function $(t, x) \mapsto A(t, x)$ returning a sparse matrix
    FMatLop(&'a dyn Fn(T, MatRef<T>) -> SparseColMat<usize, T>),
}

impl<'a, T> LinOp<T> for MatrixLinOp<'a, T>
where
    T: RealField + Float,
{
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T> {
        match self {
            MatrixLinOp::Lop(inner_lop) => inner_lop.apply_linop_to_vec(t, x, w, s),
            MatrixLinOp::MatLop(inner_lop) => {
                inner_lop * w * faer::Scale(s.unwrap_or(T::from(1.0).unwrap()))
            }
            MatrixLinOp::FMatLop(inner_lop) => {
                (inner_lop)(t, x) * w * faer::Scale(s.unwrap_or(T::from(1.0).unwrap()))
            }
        }
    }
}

/// Creates a sparse identity matrix.
///
/// # Arguments
///
/// * `dim` - the number of rows and columns
///
/// # Returns
///
/// The `dim` by `dim` sparse identity matrix.
pub fn sparse_ident<T>(dim: usize) -> SparseColMat<usize, T>
where
    T: RealField + Float,
{
    let mut ident_triplets = Vec::with_capacity(dim);
    for i in 0..dim {
        ident_triplets.push(faer::sparse::Triplet::new(i, i, T::from(1.0).unwrap()));
    }
    let ident = SparseColMat::<usize, T>::try_new_from_triplets(dim, dim, &ident_triplets).unwrap();
    ident
}

#[cfg(test)]
mod test_matexp_rs {
    // bring everything from above (parent) module into scope
    use super::*;

    /// define Lotka-Volterra system for testing ONLY
    fn lv_sys_rhs(_t: f64, x: MatRef<f64>) -> Mat<f64> {
        let alpha = 1.0;
        let beta = 1.0;
        let delta = 1.0;
        let gamma = 1.0;

        faer::mat![
            [alpha * x[(0, 0)] - beta * x[(0, 0)] * x[(1, 0)]],
            [delta * x[(0, 0)] * x[(1, 0)] - gamma * x[(1, 0)]],
        ]
    }

    /// define Lotka-Volterra jacobian for testing ONLY
    fn lv_sys_jac(_t: f64, x: MatRef<f64>) -> Mat<f64> {
        let alpha = 1.0;
        let beta = 1.0;
        let delta = 1.0;
        let gamma = 1.0;

        faer::mat![
            [alpha - beta * x[(1, 0)], -beta * x[(0, 0)]],
            [delta * x[(1, 0)], delta * x[(0, 0)] - gamma],
        ]
    }

    #[test]
    fn test_jacobian_vec_product() {
        // define x0
        let x0 = faer::mat![[1.0], [2.0],];

        // compute exact jacobian at x0
        let true_jac = lv_sys_jac(1.0, x0.as_ref());

        // compute jacobian vector product, J*w
        let w = faer::mat![[0.50], [0.75],];
        let true_jac_w = true_jac.as_ref() * w.as_ref();

        // estimate jacobian vector prod with fw finite diff
        let jac_linop = JacobianRhsLinOp::new(&lv_sys_rhs, 2);
        let approx_jac_w = jac_linop.apply_linop_to_vec(1.0, x0.as_ref(), w.as_ref(), None);

        // check
        mat_mat_approx_eq(approx_jac_w.as_ref(), true_jac_w.as_ref(), 1e-8);
    }
}
