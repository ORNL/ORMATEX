/// matrix exponential eval methods for dense faer Mats
use faer::prelude::*;
use libm::frexp;


/// Computes exp(A*dt)
pub fn matexp(a: MatRef<f64>, dt: f64) -> Mat<f64>
{
    // Calc A*t
    let a_t = a * faer::scale(dt);
    let (u, v, alpha) = matexp_pade(a_t.as_ref());
    let denom = v.as_ref() - u.as_ref();
    let numer = u.as_ref() + v.as_ref();

    // solve using QR decomp
    let denom_lu = denom.qr();
    let mut r = denom_lu.solve(numer);

    for _i in 0..alpha {
        r = r.as_ref() * r.as_ref();
    }
    r
}

/// Computes phi_k(Z)
///
/// phi_0 = exp(Z)
/// phi_1 = Z^-1 (phi_0 - I)
/// phi_2 = Z^-1 (phi_1 - (1/2!) * I)
/// phi_k = Z^-1 (phi_(k-1) - (1/k!) * I)
pub fn phi(z: MatRef<f64>, k: usize) -> Mat<f64>
{
    // phi 0 is e^Z, where Z is a matrix
    let mut phi_k = matexp(z.as_ref(), 1.0);
    if k == 0 {
        return phi_k
    }
    else {
        // let svd = faer::linalg::solvers::Svd::new(z.as_ref());
        // let z_inv = svd.pseudoinverse();
        let qr = faer::linalg::solvers::Qr::new(z.as_ref());
        let z_inv = qr.inverse();
        // identity matrix
        let id = faer::Mat::<f64>::identity(z.nrows(), z.ncols());
        for i in 1..=k {
            phi_k = z_inv.as_ref() * (phi_k.as_ref() -
                faer::scale(1.0/((1..i).product::<usize>() as f64))*id.as_ref());
        }
        phi_k
    }
}

/// Computes phi_0, phi_1, ... phi_k  together
/// using the more stable but expensive extension formula
pub fn phi_ext(z: MatRef<f64>, k: usize) -> Mat<f64>
{
    let n = z.nrows();
    let m = z.ncols();
    assert!(n == m);

    let z_ext: Mat<f64> = match k {
        0 => z.to_owned(),
        _ => {
            let z_ext_k_nrows = n+(k-1)*n;
            let z_ext_k_ncols = m;
            let mut z_ext_k = Mat::zeros(z_ext_k_nrows, z_ext_k_ncols);
            z_ext_k.get_mut(0..n, 0..m).copy_from(z);
            // z_ext_k.get_mut(n.., 0..m).copy_from(zeros);
            let z_ext_nrows = z_ext_k_nrows + n;
            let z_ext_ncols = z_ext_k_ncols + k*n;
            let mut z_ext = Mat::zeros(z_ext_nrows, z_ext_ncols);
            z_ext.get_mut(0..z_ext_k_nrows, 0..z_ext_k_ncols)
                .copy_from(z_ext_k);
            z_ext.get_mut(0..z_ext_k_nrows, z_ext_k_ncols..)
                .copy_from(Mat::<f64>::identity(k*n, k*n));
            z_ext
        }
    };

    let phi_ks = matexp(z_ext.as_ref(), 1.0);

    phi_ks.get(0..n, phi_ks.ncols()-n..).to_owned()
}

/// From N. J. Higham. The Scaling and Squaring Method for the Matrix Exponential Revisited.
/// SIAM Journal on Matrix Analysis and Applications 26.4 (2005): 1179-1193.
///
/// # Returns
/// * `U` -
/// * `V` -
/// * `alpha` - number of time the matrix is squared
pub fn matexp_pade(a: MatRef<f64>) -> (Mat<f64>, Mat<f64>, isize) {
    let mut alpha: isize = 0;
    let a_1norm = a.norm_l1();
    let a2 = a * a;
    // println!("a1_norm: {:?}", a_1norm);

    if a_1norm < 1.495585217958292e-002 {
        let (u, v) = pade3(a, a2.as_ref());
        return (u, v, alpha)
    }
    else if a_1norm < 2.539398330063230e-001{
        let a4 = a2.as_ref() * a2.as_ref();
        let (u, v) = pade5(a, a2.as_ref(), a4.as_ref());
        return (u, v, alpha)
}
    else if a_1norm < 9.504178996162932e-001{
        let a4 = a2.as_ref() * a2.as_ref();
        let a6 = a4.as_ref() * a2.as_ref();
        let (u, v) = pade7(a, a2.as_ref(), a4.as_ref(), a6.as_ref());
        return (u, v, alpha)
    }
    else if a_1norm < 2.097847961257068e+000{
        let a4 = a2.as_ref() * a2.as_ref();
        let a6 = a4.as_ref() * a2.as_ref();
        let a8 = a6.as_ref() * a2.as_ref();
        let (u, v) = pade9(a, a2.as_ref(), a4.as_ref(), a6.as_ref(), a8.as_ref());
        return (u, v, alpha)
    }
    else {
        let maxnorm = 5.371920351148152;
        let (_m, _a) = frexp(a_1norm / maxnorm);
        alpha = _a as isize;
        if alpha < 0 {
            alpha = 0;
        }
        let scale = (2.0 as f64).powf(alpha as f64);
        let a_scaled = a * faer::scale(1. / scale as f64);
        let a2_scaled = a_scaled.as_ref() * a_scaled.as_ref();
        let a4_scaled = a2_scaled.as_ref() * a2_scaled.as_ref();
        let a6_scaled = a4_scaled.as_ref() * a2_scaled.as_ref();
        let (u, v) = pade13(a_scaled.as_ref(), a2_scaled.as_ref(), a4_scaled.as_ref(), a6_scaled.as_ref());
        return (u, v, alpha)
    }
}

/// Private pade methods impl
///
/// # Args
/// * `a` : sparse matrix
/// * `a2` : sparse matrix squared
///
/// # Returns
/// * `U` -
/// * `V` -
fn pade3(a: MatRef<f64>, a2: MatRef<f64>)
        -> (Mat<f64>, Mat<f64>) {
    const B3: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
    let ident: Mat<f64> = faer::Mat::identity(a.ncols(), a.nrows());
    let temp = a2 * faer::scale(B3[3]) + ident.as_ref() * faer::scale(B3[1]);
    let u = a * temp;
    let v = a2 * faer::scale(B3[2]) + ident.as_ref() * faer::scale(B3[0]);
    (u, v)
}
fn pade5(a: MatRef<f64>, a2: MatRef<f64>, a4: MatRef<f64>)
        -> (Mat<f64>, Mat<f64>) {
    const B5: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
    let ident: Mat<f64> = faer::Mat::identity(a.ncols(), a.nrows());
    let temp = a4 * faer::scale(B5[5]) + a2 * faer::scale(B5[3]) + ident.as_ref()*faer::scale(B5[1]);
    let u = a * temp;
    let v = a4 * faer::scale(B5[4]) + a2 * faer::scale(B5[2]) + ident.as_ref() * faer::scale(B5[0]);
    (u, v)
}
fn pade7(a: MatRef<f64>, a2: MatRef<f64>, a4: MatRef<f64>, a6: MatRef<f64>)
        -> (Mat<f64>, Mat<f64>) {
    const B7: [f64; 8] = [17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.];
    let ident: Mat<f64> = faer::Mat::identity(a.ncols(), a.nrows());
    let temp = a6 * faer::scale(B7[7]) + a4 * faer::scale(B7[5]) +
        a2 * faer::scale(B7[3]) + ident.as_ref() * faer::scale(B7[1]);
    let u = a * temp;
    let v = a6 * faer::scale(B7[6]) + a4 * faer::scale(B7[4]) + a2 * faer::scale(B7[2]) +
        ident.as_ref() * faer::scale(B7[0]);
    (u, v)
}
fn pade9(a: MatRef<f64>, a2: MatRef<f64>, a4: MatRef<f64>, a6: MatRef<f64>, a8: MatRef<f64>)
        -> (Mat<f64>, Mat<f64>) {
    const B9: [f64; 10] = [17643225600., 8821612800., 2075673600.,
                           302702400.,   30270240.,   2162160.,
                           110880.,      3960.,       90.,
                           1.];
    let ident: Mat<f64> = faer::Mat::identity(a.ncols(), a.nrows());
    let temp = a8 * faer::scale(B9[9]) + a6 * faer::scale(B9[7]) + a4*faer::scale(B9[5])
        + a2*faer::scale(B9[3]) + ident.as_ref() * faer::scale(B9[1]);
    let u = a * temp;
    let v = a8 * faer::scale(B9[8]) + a6 * faer::scale(B9[6]) + a4*faer::scale(B9[4]) +
        a2*faer::scale(B9[2]) + ident.as_ref() * faer::scale(B9[0]);
    (u, v)
}
fn pade13(a: MatRef<f64>, a2: MatRef<f64>, a4: MatRef<f64>, a6: MatRef<f64>)
        -> (Mat<f64>, Mat<f64>) {
    // Pade Constants
    const B13: [f64; 14] = [64764752532480000., 32382376266240000., 7771770303897600.,
                            1187353796428800.,  129060195264000.,   10559470521600.,
                            670442572800.,      33522128640.,       1323241920.,
                            40840800.,          960960.,            16380.,
                            182.,               1.];
    let ident: Mat<f64> = faer::Mat::identity(a.ncols(), a.nrows());
    let v1 = a6*faer::scale(B13[13]) + a4*faer::scale(B13[11]) + a2*faer::scale(B13[9]);
    let temp = a6 * v1.as_ref() +
        a6*faer::scale(B13[7]) + a4*faer::scale(B13[5]) + a2*faer::scale(B13[3]) +
        ident.as_ref()*faer::scale(B13[1]);
    let u = a * temp;
    let temp2 = a6*faer::scale(B13[12]) + a4*faer::scale(B13[10]) + a2*faer::scale(B13[8]);
    let v2 = a6 * temp2 +
        a6*faer::scale(B13[6]) + a4*faer::scale(B13[4]) + a2*faer::scale(B13[2]) +
        ident.as_ref()*faer::scale(B13[0]);
    (u, v2)
}


#[cfg(test)]
mod test_matexp_pade {
    use crate::mat_utils::{random_mat_normal, mat_mat_approx_eq};

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_phi_ext() {
        // ensure the phi extension formula produces same as recurrance formula
        // contruct a random mat
        let dense_a: Mat<f64> = random_mat_normal(5, 5);

        for k in 0 ..= 3 {
            let phi_a = phi(dense_a.as_ref(), k);
            let phi_ext_a = phi_ext(dense_a.as_ref(), k);
            mat_mat_approx_eq(phi_a.as_ref(), phi_ext_a.as_ref(), 1e-10);
        }
    }

}
