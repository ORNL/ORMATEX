use std::cell::RefCell;
use faer::prelude::*;
use faer::sparse::*;
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

// Helper function to ensure two matrix are almost equal
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


/// Arnoldi iteration
///
/// This process produces an orthonomal basis Q and an
/// upper Hessinberg matrix, H.
///
/// # Arguments
/// * `a`: Sparse input matrix
/// * `b`: vector used to build the subspace
/// * `n`: desired krylov dimension
pub fn arnoldi<T>(a: SparseColMatRef<usize, T>, b: MatRef<T>, n: usize) -> (Mat<T>, SparseColMat<usize, T>)
    where
    T: RealField + Float
{
    let mut hs = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    let q0 = b.to_owned() * faer::Scale(T::from(1.0).unwrap() / b.norm_l2());
    vs.push(q0);

    for k in 0..n {
        let (hk, qk) = arnoldi_inner(a, &vs, k, n);
        hs.push(hk);
        if k+1 < n {
            vs.push(qk);
        }
    }

    // build full sparse H matrix from column vecs
    // create sparse matrix from triplets
    let mut h_triplets = Vec::new();
    let mut h_len = 0;
    for (c, hvec) in (&hs).into_iter().enumerate() {
        h_len = hvec.len();
        for h_i in 0..h_len {
            h_triplets.push(faer::sparse::Triplet::new(h_i, c, hvec[h_i]));
        }
    }
    let h_sprs = SparseColMat::<usize, T>::try_new_from_triplets(
        h_len, (&hs).len(), &h_triplets).unwrap();


    // build full Q matrix
    let mut q_out: Mat<T> = faer::Mat::zeros(vs[0].nrows(), vs.len());
    for j in 0..q_out.ncols() {
        for i in 0..q_out.nrows() {
            q_out[(i, j)] = vs[j][(i, 0)];
        }
    }

    (q_out, h_sprs)
}

fn arnoldi_inner<T>(
    a: SparseColMatRef<usize, T>,
    q: &Vec<Mat<T>>,
    k: usize,
    n: usize,
) -> (Vec<T>, Mat<T>)
    where
    T: RealField + Float
{
    // Krylov vector
    let q_col: MatRef<T> = q[k].as_ref();

    // If A is a Jacobian, a Jacobian-vector product can be
    // given as $`J q \approx (F(x + \eps q) - F(x)) / \eps `$
    let mut qv: Mat<T> = a * q_col;

    let mut h = Vec::with_capacity(k + 2);
    for i in 0..=k {
        let qci: MatRef<T> = q[i].as_ref();
        let ht = qv.transpose() * qci;
        h.push( ht[(0, 0)] );
        qv = qv - (qci * faer::Scale(h[i]));
    }
    if k+1 < n {
        let norm_v = qv.norm_l2();
        // println!("norm v={:?}", &norm_v);
        h.push(norm_v);
        qv = qv * faer::Scale(T::from(1.).unwrap()/h[k + 1]);
    }
    return (h, qv);
}


/// Arnoldi iteration with linear operator A
pub fn arnoldi_lo<T>(a_lo: &MatrixLinOp<T>, t: T, a_lo_scale: T, b: MatRef<T>, n: usize) -> (Mat<T>, SparseColMat<usize, T>, i32)
    where
    T: RealField + Float
{
    let mut hs = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    let q0 = b.to_owned() * faer::Scale(T::from(1.0).unwrap() / b.norm_l2());
    vs.push(q0);

    let mut breakdown_n = -1;

    for k in 0..n {
        let h_q_k = arnoldi_inner_lo(a_lo, t, a_lo_scale, b.as_ref(), &vs, k, n);
        match h_q_k {
            Ok(h_q_k) => {
                let hk = h_q_k.0;
                let qk = h_q_k.1;

                hs.push(hk);
                if k+1 < n {
                    vs.push(qk);
                }
            },
            Err(h_q_k) => {
                // breakdown in arnoldi, q vec is no good
                let hk = h_q_k;
                hs.push(hk);
                breakdown_n = k as i32;
                break;
                // if k+1 < n {
                //     vs.push(b.as_ref() * faer::Scale(T::from(0.0).unwrap()));
                // }
            }
        }
    }

    // build full sparse H matrix from column vecs
    // create sparse matrix from triplets
    let mut h_triplets = Vec::new();
    let mut h_len = 0;
    for (c, hvec) in (&hs).into_iter().enumerate() {
        h_len = hvec.len();
        for h_i in 0..h_len {
            h_triplets.push(faer::sparse::Triplet::new(h_i, c, hvec[h_i]));
        }
    }
    let h_sprs = SparseColMat::<usize, T>::try_new_from_triplets(
        h_len, (&hs).len(), &h_triplets).unwrap();
    // check that H is square
    if h_len != (&hs).len() {
        println!("{:?}", hs);
    }
    assert!(h_len > 0);
    assert!((&hs).len() > 0);
    assert!(h_len == (&hs).len());

    // build full Q matrix
    let mut q_out: Mat<T> = faer::Mat::zeros(vs[0].nrows(), vs.len());
    for j in 0..q_out.ncols() {
        for i in 0..q_out.nrows() {
            q_out[(i, j)] = vs[j][(i, 0)];
        }
    }

    (q_out, h_sprs, breakdown_n)
}

/// Arnoldi inner iteration with linear operator A
fn arnoldi_inner_lo<T>(
    a_lo: &MatrixLinOp<T>,
    t: T,
    a_lo_scale: T,
    v0: MatRef<T>,
    q: &Vec<Mat<T>>,
    k: usize,
    n: usize,
) -> Result<(Vec<T>, Mat<T>), Vec<T>>
    where
    T: RealField + Float
{
    // breakdown tol
    let eps = T::from(1e-14).unwrap();

    // Krylov vector
    let q_col: MatRef<T> = q[k].as_ref();

    // let mut qv: Mat<T> = a * q_col;
    let mut qv: Mat<T> = a_lo.apply_linop_to_vec(t, v0, q_col.as_ref(), Some(a_lo_scale));
    // let mut qv: Mat<T> = a_lo.apply_linop(a_lo, q_col.as_ref()) * a_lo_scale;

    let mut h = Vec::with_capacity(k + 2);
    for i in 0..=k {
        let qci: MatRef<T> = q[i].as_ref();
        let ht = qv.transpose() * qci;
        h.push( ht[(0, 0)] );
        qv = qv - (qci * faer::Scale(h[i]));
    }
    let norm_v = qv.norm_l2();
    if k+1 < n && norm_v >= eps {
        h.push(norm_v);
        // if norm_v is zero this is a div by 0 err
        qv = qv * faer::Scale(T::from(1.).unwrap()/norm_v);
    }
    // breakdown, qv is the zero vector
    if norm_v < eps {
        return Err(h);
    }
    return Ok((h, qv));
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

    #[test]
    fn test_arnoldi() {
        // test matrix
        let dense_a: Mat<f64> = random_mat_normal(10, 10);
        let test_a = dense_to_sprs(dense_a.as_ref());

        // pick a starting vector and normalize it
        let mut q0: Mat<f64> = random_mat_normal(10, 1);
        q0 = q0.as_ref() * faer::Scale(1.0 / q0.norm_l2());

        // arnoldi
        let (q, h) = arnoldi(test_a.as_ref(), q0.as_ref(), 10);

        println!("{:?}", q);
        println!("{:?}", h.to_dense());

        // ensure Q is orthonormal
        let qt_q = q.as_ref().transpose() * q.as_ref();
        mat_mat_approx_eq(qt_q.as_ref(), faer::Mat::identity(10, 10).as_ref(), 1.0e-12);

        println!("q shape = {:?}, {:?}", q.nrows(), q.ncols());
        println!("h shape = {:?}, {:?}", h.nrows(), h.ncols());

        // check that Q^T*A*Q = H
        let h_test = (q.as_ref().transpose() * test_a.as_ref() * q.as_ref() - h.as_ref().to_dense()).norm_l2()
            * (1. / test_a.to_dense().norm_l2());
        assert_approx_eq!(h_test, 0.0, 1.0e-12);
    }

    #[test]
    fn test_arnoldi_breakdown() {
        // test arnoldi when breakdown (div by zero) in the method is expected
        // columns are not linear indep
        let dense_a: Mat<f64> = faer::mat![
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ];
        let test_a = dense_to_sprs(dense_a.as_ref());

        // pick a starting vector and normalize it
        let mut q0: Mat<f64> = random_mat_normal(5, 1);
        q0 = q0.as_ref() * faer::Scale(1.0 / q0.norm_l2());

        // arnoldi
        let (q, h) = arnoldi(test_a.as_ref(), q0.as_ref(), 5);
        println!("{:?}", q.as_ref());
        println!("{:?}", h.to_dense().as_ref());

        // construct liner operator from a mat
        let test_a_lop = MatrixLinOp::MatLop(test_a.as_ref());

        // arnoldi with linear op
        let (q_lo, h_lo, brkdwn) = arnoldi_lo(&test_a_lop, 1.0, 1.0, q0.as_ref(), 5);
        println!("{:?}", brkdwn);
        println!("{:?}", q_lo.as_ref());
        println!("{:?}", h_lo.to_dense().as_ref());

        assert!(brkdwn >= 0);
    }

    #[test]
    fn test_arnoldi_sprs() {
        // test matrix
        let dense_a: Mat<f64> = faer::mat![
            [2.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 2.0],
        ];
        let test_a = dense_to_sprs(dense_a.as_ref());

        // pick a starting vector and normalize it
        let mut q0: Mat<f64> = faer::mat![
            [2.0,],
            [1.0,],
            [0.0,],
            [0.0,],
            [0.0,],
        ];
        q0 = q0.as_ref() * faer::Scale(1.0 / q0.norm_l2());

        // arnoldi
        let (q, h) = arnoldi(test_a.as_ref(), q0.as_ref(), 5);

        println!("{:?}", q);
        println!("{:?}", h.to_dense());

        // ensure Q is orthonormal
        let qt_q = q.as_ref().transpose() * q.as_ref();
        mat_mat_approx_eq(qt_q.as_ref(), faer::Mat::identity(5, 5).as_ref(), 1.0e-12);

        println!("q shape = {:?}, {:?}", q.nrows(), q.ncols());
        println!("h shape = {:?}, {:?}", h.nrows(), h.ncols());

        // check that Q^T*A*Q = H
        let h_test = (q.as_ref().transpose() * test_a.as_ref() * q.as_ref() - h.as_ref().to_dense()).norm_l2()
            * (1. / test_a.to_dense().norm_l2());
        assert_approx_eq!(h_test, 0.0, 1.0e-12);
    }

    #[test]
    fn test_arnoldi_lo() {
        // test arnoldi with linear operator for A
        let dense_a: Mat<f64> = random_mat_normal(10, 10);
        let test_a = dense_to_sprs(dense_a.as_ref());

        // pick a starting vector and normalize it
        let mut q0: Mat<f64> = random_mat_normal(10, 1);
        q0 = q0.as_ref() * faer::Scale(1.0 / q0.norm_l2());

        // base arnoldi
        let (q_base, h_base) = arnoldi(test_a.as_ref(), q0.as_ref(), 10);

        // construct liner operator from a mat
        let test_a_lop = MatrixLinOp::MatLop(test_a.as_ref());

        // arnoldi with linear op
        let (q, h, brkdwn) = arnoldi_lo(&test_a_lop, 1.0, 1.0, q0.as_ref(), 10);
        // brkdwn flag < 0 means method terminated without breakdown (no div by zero detected)
        // assert!(brkdwn < 0);
        mat_mat_approx_eq(q_base.as_ref(), q.as_ref(), 1.0e-8);
        mat_mat_approx_eq(h_base.to_dense().as_ref(), h.to_dense().as_ref(), 1.0e-8);

        // ensure Q is orthonormal
        let qt_q = q.as_ref().transpose() * q.as_ref();
        mat_mat_approx_eq(qt_q.as_ref(), faer::Mat::identity(10, 10).as_ref(), 1.0e-12);

        println!("q shape = {:?}, {:?}", q.nrows(), q.ncols());
        println!("h shape = {:?}, {:?}", h.nrows(), h.ncols());

        // check that Q^T*A*Q = H
        let h_test = (q.as_ref().transpose() * test_a.as_ref() * q.as_ref() - h.as_ref().to_dense()).norm_l2()
            * (1. / test_a.to_dense().norm_l2());
        assert_approx_eq!(h_test, 0.0, 1.0e-12);

    }

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
