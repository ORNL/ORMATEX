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
use std::cell::RefCell;
use faer::prelude::*;
use faer_traits::ComplexField;
use faer_traits::RealField;
use num_traits::Float;
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};


/// create a matrix filled with standard normal samples
pub fn random_mat_normal<T>(n_rows: usize, n_cols: usize)
    -> Mat<T>
    where
    T: RealField + Float
{
    let omega: Mat<T> = Mat::from_fn(
        n_rows,
        n_cols,
        |_i, _j| {
            T::from::<f64>(
            thread_rng().sample(StandardNormal)).unwrap()
            }
        );
    omega
}

/// create a matrix filled with uniform random samples
pub fn random_mat_uniform<T>(n_rows: usize, n_cols: usize, lb: f64, ub: f64)
    -> Mat<T>
    where
    T: RealField + Float
{
    let uni_dist = Uniform::new(lb, ub);
    let omega: Mat<T> = Mat::from_fn(
        n_rows,
        n_cols,
        |_i, _j| {
            T::from::<f64>(
            thread_rng().sample(uni_dist)).unwrap()
            }
        );
    omega
}

/// Helper function to ensure two matrix are almost equal
pub fn mat_mat_approx_eq<T>(a: MatRef<T>, b: MatRef<T>, tol: T)
    where
    T: RealField + Float
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

/// Only the real part of mat
pub fn real_mat(a: MatRef<c64>) -> Mat<f64> {
    let rm: Mat<f64> = Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
            a[(i, j)].re
        }
    );
    rm
}

/// Convert mat to complex and scale by dt
pub fn complex_mat_scale(a: MatRef<f64>, dt: f64) -> Mat<c64>
{
    let a_dt: Mat<c64> = Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
        c64::from( a[(i, j)] ) * c64::from(dt)
        }
    );
    a_dt
}

/// Take powers of a real matrix
pub fn mat_pow<T>(a: MatRef<T>, p: usize) -> Mat<T>
    where
    T: ComplexField
{
    let mut ap_out: Mat<T> = Mat::identity(a.nrows(), a.ncols());
    for _i in 0..p {
        ap_out = a.as_ref() * ap_out.as_ref();
    }
    ap_out
}

// Helper function to convert a dense mat to a sparse mat.
// For testing ONLY
pub fn dense_to_sprs<T>(a: MatRef<T>) -> SparseColMat<usize, T>
    where
    T: RealField + Float
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
    let out = SparseColMat::<usize, T>::try_new_from_triplets(a.nrows(), a.ncols(), &a_triplets).unwrap();
    out
}

/// Linear Operator
pub trait LinOp<T>
    where
    T: RealField + Float
{
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T>;
}

/// If A is a Jacobian, a Jacobian-vector product can be
/// given as $`A q \approx (F(x + \eps w) - F(x)) / \eps `$
/// where $`F`$ is `frhs`
#[derive(Clone)]
pub struct JacobianRhsLinOp<'a, T>
    where
    T: RealField + Float
{
    /// Function ref to RHS of the system
    frhs: &'a dyn Fn(T, MatRef<T>) -> Mat<T>,

    /// Pointer to storage for x vector cache
    x_tmp: RefCell<Mat<T>>,

    /// Pointer to storage for F(x) vector (RHS eval) cache
    fx_tmp: RefCell<Mat<T>>,
}

impl <'a, T> LinOp<T> for JacobianRhsLinOp<'a, T>
    where
    T: RealField + Float
{
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T> {
        let x_norm_l1 = x.norm_l1();
        if x_norm_l1 == self.x_tmp.borrow().as_ref().norm_l1() {
            // we can reuse prior frhs eval
        }
        else {
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
        let Jw: Mat<T> = faer::Scale(scaler) * ((self.frhs)(t, x_pert.as_ref()) - (self.fx_tmp.borrow().as_ref()))
            * faer::Scale(ieps);
        Jw
    }
}
impl <'a, T> JacobianRhsLinOp <'a, T>
    where
    T: RealField + Float
{
    pub fn new(frhs: &'a dyn Fn(T, MatRef<T>) -> Mat<T>, dim: usize) -> Self {
        Self {
            frhs,
            x_tmp: RefCell::new(faer::Mat::zeros(dim, dim)),
            fx_tmp: RefCell::new(faer::Mat::zeros(dim, dim)),
        }
    }
}

/// Wrapper around a sparse matrix ref to apply it to a vec
pub struct JacobianMatLinOp<'a, T>
    where
    T: RealField + Float
{
    a_mat: SparseColMatRef<'a, usize, T>,
}
impl <'a, T> JacobianMatLinOp <'a, T>
    where
    T: RealField + Float
{
    pub fn new(a_mat: SparseColMatRef<'a, usize, T>) -> Self {
        Self {
            a_mat,
        }
    }
}
impl <'a, T> LinOp<T> for JacobianMatLinOp<'a, T>
    where
    T: RealField + Float
{
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T> {
        self.a_mat * w * faer::Scale(s.unwrap_or(T::from(1.0).unwrap()))
    }
}

/// Enum of linear operators
#[derive(Clone)]
pub enum MatrixLinOp<'a, T>
    where
    T: RealField + Float
{
    Lop(&'a dyn LinOp<T>),
    MatLop(SparseColMatRef<'a, usize, T>),
    FMatLop(&'a dyn Fn(T, MatRef<T>) -> SparseColMat<usize, T>),
}

impl <'a, T> LinOp<T> for MatrixLinOp<'a, T>
    where
    T: RealField + Float
{
    fn apply_linop_to_vec(&self, t: T, x: MatRef<T>, w: MatRef<T>, s: Option<T>) -> Mat<T> {
        match self {
            MatrixLinOp::Lop(inner_lop) => inner_lop.apply_linop_to_vec(t, x, w, s),
            MatrixLinOp::MatLop(inner_lop) => inner_lop * w * faer::Scale(s.unwrap_or(T::from(1.0).unwrap())),
            MatrixLinOp::FMatLop(inner_lop) => (inner_lop)(t, x) * w * faer::Scale(s.unwrap_or(T::from(1.0).unwrap()))
        }
    }
}


/// sparse identity
pub fn sparse_ident<T>(dim: usize) -> SparseColMat<usize, T>
    where
    T: RealField + Float
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
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

    /// define Lotka-Volterra system for testing ONLY
    fn lv_sys_rhs(t: f64, x: MatRef<f64>) -> Mat<f64> {
        let alpha = 1.0;
        let beta = 1.0;
        let delta = 1.0;
        let gamma = 1.0;

        faer::mat![
            [alpha * x[(0, 0)] - beta * x[(0, 0)]*x[(1, 0)] ],
            [delta * x[(0, 0)]*x[(1, 0)] - gamma * x[(1, 0)] ],
        ]
    }

    /// define Lotka-Volterra jacobian for testing ONLY
    fn lv_sys_jac(t: f64, x: MatRef<f64>) -> Mat<f64> {
        let alpha = 1.0;
        let beta = 1.0;
        let delta = 1.0;
        let gamma = 1.0;

        faer::mat![
            [alpha - beta*x[(1, 0)], -beta*x[(0, 0)] ],
            [delta*x[(1, 0)], delta*x[(0, 0)] - gamma ],
        ]
    }

    #[test]
    fn test_jacobian_vec_product() {

        // define x0
        let x0 = faer::mat![
            [1.0],
            [2.0],
        ];

        // compute exact jacobian at x0
        let true_jac = lv_sys_jac(1.0, x0.as_ref());

        // comput jacobian vector product, J*w
        let w = faer::mat![
            [0.50],
            [0.75],
        ];
        let true_jac_w = true_jac.as_ref() * w.as_ref();

        // estimate jacobian vector prod with fw finite diff
        let mut jac_linop = JacobianRhsLinOp::new(&lv_sys_rhs, 2);
        let approx_jac_w = jac_linop.apply_linop_to_vec(1.0, x0.as_ref(), w.as_ref(), None);

        // check
        mat_mat_approx_eq(approx_jac_w.as_ref(), true_jac_w.as_ref(), 1e-8);
    }
}
