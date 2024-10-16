/// Contains arnoldi iteration methods
/// Provides arnoldi methods for both faer LinOp and faer SparseColMat
use faer::prelude::*;
use faer::sparse::*;
use faer::linop::LinOp;
use reborrow::ReborrowMut;
use std::cmp;
use faer::dyn_stack::PodStack;
use num_traits::Float;


/// Arnoldi inner iteration with linear operator A
///
/// #Args
/// * `a_lo` - linear operator, sparse mat or method to apply mat to vec
/// * `a_lo_scale` - scale factor on the linear operator
/// * `k` - current krylov iteration
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth
/// * `hs` - upper hessenberg
/// * `qs` - orthonormal basis of kyrlov subspace
fn arnoldi_inner_lop<T, L>(
    a_lo: L,
    a_lo_scale: T,
    k: usize,
    n: usize,
    iom: usize,
    hs: MatMut<T>,
    mut qs: MatMut<T>
) -> bool
    where
    T: faer::RealField + Float,
    L: LinOp<T>
{
    // dummy
    let mut _dummy_podstack: [u8;1] = [0u8;1];

    // incomplete orth depth
    let iom_depth = cmp::max(k as i32 - iom as i32 , 0) as usize;

    // breakdown tol
    let eps = T::from(1e-14).unwrap();

    // Krylov vector
    let q_col: ColRef<T> = qs.rb_mut().col(k);

    // let mut qv: Mat<T> = a_lo * q_col;
    let mut qv: Mat<T> = faer::Mat::zeros(q_col.nrows(), 1);
    a_lo.apply(qv.as_mut(),
               q_col.as_2d().as_ref(),
               faer::get_global_parallelism(),
               PodStack::new(&mut _dummy_podstack));
    qv = qv * faer::scale(a_lo_scale);

    // let mut h = Vec::with_capacity(k + 2);
    // let mut h = vec![T::from(0.0).unwrap(); k+2];
    let mut h = hs.col_mut(k);
    for i in iom_depth..=k {
        let qci: ColRef<T> = qs.rb_mut().col(i);
        let ht = qv.col(0).transpose() * qci;
        h.write(i, ht);
        qv = qv - (qci.as_2d() * faer::scale(ht));
    }

    let norm_v = qv.norm_l2();
    if k+1 < n {
        h.write(k+1, norm_v);
    }

    if k+1 < n && norm_v >= eps {
        // if norm_v is zero this is a div by 0 err
        qv = qv * faer::scale(T::from(1.).unwrap()/norm_v);
        qs.col_mut(k+1).copy_from(qv.col(0));
    }
    // breakdown, qv is the zero vector
    if norm_v < eps {
        return true
    }
    return false
}


/// Arnoldi iteration with linear operator A
///
/// #Args
/// * `a_lo` - linear operator, sparse mat or method to apply mat to vec
/// * `a_lo_scale` - scale factor on the linear operator
/// * `b` - initial vector in [b, Ab, A^2b, ...]
/// * `n` - max krylov iteration
/// * `iom` - incomplete ortho depth
pub fn arnoldi_lop<T, L>(
    a_lo: L,
    a_lo_scale: T,
    b: MatRef<T>,
    n: usize,
    iom: usize
) -> (Mat<T>, Mat<T>, usize)
    where
    T: faer::RealField + Float,
    L: LinOp<T>
{
    let mut hs = faer::Mat::zeros(n, n);
    let mut qs = faer::Mat::zeros(b.nrows(), n);
    let q0 = b * faer::scale(T::from(1.0).unwrap() / b.norm_l2());
    qs.col_mut(0).copy_from(q0.col(0));

    let mut breakdown_n = 0;

    for k in 0..n {
        let breakdown_flag = arnoldi_inner_lop(
            &a_lo, a_lo_scale, k, n, iom, hs.as_mut(), qs.as_mut());
        breakdown_n += 1;
        match breakdown_flag {
            false => {},
            true => {
                break;
            }
        }
    }

    (
        qs.get(0..b.nrows(), 0..breakdown_n).to_owned(),
        hs.get(0..breakdown_n, 0..breakdown_n).to_owned(),
        breakdown_n
    )
}


#[cfg(test)]
mod test_arnoldi {
    use assert_approx_eq::assert_approx_eq;
    use faer::assert_matrix_eq;
    use crate::mat_utils::{dense_to_sprs, random_mat_normal, arnoldi, mat_mat_approx_eq};

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_arnoldi_lop_dens() {
        // test that arnoldi works with a dense matrix
    }

    #[test]
    fn test_arnoldi_lop_sprs() {
        // test that arnoldi works with a sparse matrix
        // test arnoldi with linear operator for A
        let dense_a: Mat<f64> = random_mat_normal(10, 10);
        let test_a = dense_to_sprs(dense_a.as_ref());

        // pick a starting vector and normalize it
        let mut q0: Mat<f64> = random_mat_normal(10, 1);
        q0 = q0.as_ref() * faer::scale(1.0 / q0.norm_l2());

        // base arnoldi
        let (q_base, h_base) = arnoldi(test_a.as_ref(), q0.as_ref(), 10);
        let qt_q_base = q_base.as_ref().transpose() * q_base.as_ref();
        mat_mat_approx_eq(qt_q_base.as_ref(), faer::Mat::identity(10, 10).as_ref(), 1.0e-12);
        println!("base arnoldi: \n {:?}", q_base);

        // arnoldi with linear op
        let iom = 1000;
        let kd = 10;
        let (q, h, brkdwn) = arnoldi_lop(test_a.as_ref(), 1.0, q0.as_ref(), kd, iom);
        println!("arnoldi linop: \n {:?}", q);
        // brkdwn flag < 0 means method terminated without breakdown (no div by zero detected)
        // assert!(brkdwn < 0);
        mat_mat_approx_eq(q_base.as_ref(), q.as_ref(), 1.0e-8);
        mat_mat_approx_eq(h_base.to_dense().as_ref(), h.as_ref(), 1.0e-8);

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
}
