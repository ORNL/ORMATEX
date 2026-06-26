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
/// Matrix exponential evaluation methods for dense faer Mats.
///
/// All public functions are generic over `T: ComplexField` (covers both `f64`
/// and `c64 = Complex<f64>`). The time step `dt` is always a real `f64`,
/// consistent with the assumption that time is a real quantity.
use faer::prelude::*;
use faer::linalg::solvers::{Solve, DenseSolveCore};
use faer::complex::ComplexFloat;
use faer_traits::ComplexField;
use faer_traits::math_utils::from_f64;
use num_traits::ToPrimitive;
use statrs::function::factorial;
use crate::matexp_traits::DensePhikvEvaluator;
use libm::frexp;


#[derive(Debug)]
pub struct PadeExpm {
    max_squarings: usize,
}

impl PadeExpm {
    pub fn new(max_squarings: usize) -> Self
    {
        Self {
            max_squarings
        }
    }
}

/// Generic implementation — works for both `f64` and `c64` matrices.
/// `dt` is kept as `f64` (real time step).
impl<T> DensePhikvEvaluator<T> for PadeExpm
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    fn phik_apply(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>, k: usize) -> Mat<T> {
        phi_ext((Scale(from_f64::<T>(dt)) * a).as_ref(), k) * v0
    }
}

/// Computes exp(A * dt) for real or complex matrix `A`.
///
/// `dt` is a real scalar time step.
pub fn matexp<T>(a: MatRef<T>, dt: f64) -> Mat<T>
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    let a_t = a * Scale(from_f64::<T>(dt));
    let (u, v, alpha) = matexp_pade(a_t.as_ref());
    let denom = v.as_ref() - u.as_ref();
    let numer = u.as_ref() + v.as_ref();

    // Solve (V - U) \ (V + U) via QR decomposition
    let denom_qr = denom.qr();
    let mut r = denom_qr.solve(numer);

    for _i in 0..alpha {
        r = r.as_ref() * r.as_ref();
    }
    r
}

/// Computes phi_k(Z) for a square matrix `Z` via the recurrence relation.
///
/// The phi functions satisfy the recurrence (derived from the series definition):
/// ```text
/// phi_0(Z) = exp(Z)
/// phi_k(Z) = Z^{-1} (phi_{k-1}(Z) - 1/(k-1)! · I)    for k ≥ 1
/// ```
/// Note the `(k-1)!` factor — not `k!`.  At step k=1 this subtracts `1/0! = 1`,
/// at k=2 it subtracts `1/1! = 1`, at k=3 it subtracts `1/2! = 0.5`, etc.
///
/// For better numerical stability, prefer [`phi_ext`] which uses the
/// augmented-matrix formula.
pub fn phi<T>(z: MatRef<T>, k: usize) -> Mat<T>
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    let mut phi_k = matexp(z.as_ref(), 1.0);
    if k == 0 {
        return phi_k
    }
    let qr = faer::linalg::solvers::Qr::new(z.as_ref());
    let z_inv = qr.inverse();
    let id = Mat::<T>::identity(z.nrows(), z.ncols());
    for i in 1..=k {
        // the phi recurrence: phi_k = Z^{-1}(phi_{k-1} - 1/(k-1)! · I).
        let fact = (1..i).product::<usize>() as f64;
        let fact = if fact == 0.0 { 1.0 } else { fact };
        phi_k = z_inv.as_ref() * (phi_k.as_ref()
            - Scale(from_f64::<T>(1.0 / fact)) * id.as_ref());
    }
    phi_k
}

/// Computes phi_k(Z) using the numerically stable augmented-matrix formula.
///
/// Constructs the block extension
/// ```text
/// Z_ext = [ Z | I_k ]
///         [ 0 | K  ]
/// ```
/// of size `n(k+1) × n(k+1)`, computes `exp(Z_ext)`, and extracts the
/// top-right `n × n` block which equals `phi_k(Z)`.
pub fn phi_ext<T>(z: MatRef<T>, k: usize) -> Mat<T>
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    let n = z.nrows();
    let m = z.ncols();
    assert_eq!(n, m, "phi_ext requires a square matrix");

    let z_ext: Mat<T> = match k {
        0 => z.to_owned(),
        _ => {
            let z_ext_k_nrows = n + (k - 1) * n;
            let z_ext_k_ncols = m;
            let z_ext_nrows = z_ext_k_nrows + n;
            let z_ext_ncols = z_ext_k_ncols + k * n;
            let mut z_ext = Mat::<T>::zeros(z_ext_nrows, z_ext_ncols);
            z_ext.get_mut(0..n, 0..m)
                .copy_from(z);
            z_ext.get_mut(0..z_ext_k_nrows, z_ext_k_ncols..)
                .copy_from(Mat::<T>::identity(k * n, k * n));
            z_ext
        }
    };

    let phi_ks = matexp(z_ext.as_ref(), 1.0);
    phi_ks.get(0..n, phi_ks.ncols() - n..).to_owned()
}

/// Selects and applies the cheapest Padé approximant to `A` based on its
/// 1-norm, following:
///
/// > N. J. Higham. *The Scaling and Squaring Method for the Matrix Exponential
/// > Revisited.* SIAM J. Matrix Anal. Appl. 26(4):1179–1193, 2005.
///
/// # Returns
/// `(U, V, alpha)` where `exp(A) ≈ (V - U)⁻¹(V + U)` after squaring
/// `alpha` times.
pub fn matexp_pade<T>(a: MatRef<T>) -> (Mat<T>, Mat<T>, isize)
where
    T: ComplexField,
    T::Real: ToPrimitive,
{
    let mut alpha: isize = 0;
    let a_1norm = a.norm_l1();
    let a2 = a * a;

    if a_1norm < from_f64::<T::Real>(1.495585217958292e-002) {
        let (u, v) = pade3(a, a2.as_ref());
        return (u, v, alpha)
    }
    else if a_1norm < from_f64::<T::Real>(2.539398330063230e-001) {
        let a4 = a2.as_ref() * a2.as_ref();
        let (u, v) = pade5(a, a2.as_ref(), a4.as_ref());
        return (u, v, alpha)
    }
    else if a_1norm < from_f64::<T::Real>(9.504178996162932e-001) {
        let a4 = a2.as_ref() * a2.as_ref();
        let a6 = a4.as_ref() * a2.as_ref();
        let (u, v) = pade7(a, a2.as_ref(), a4.as_ref(), a6.as_ref());
        return (u, v, alpha)
    }
    else if a_1norm < from_f64::<T::Real>(2.097847961257068e+000) {
        let a4 = a2.as_ref() * a2.as_ref();
        let a6 = a4.as_ref() * a2.as_ref();
        let a8 = a6.as_ref() * a2.as_ref();
        let (u, v) = pade9(a, a2.as_ref(), a4.as_ref(), a6.as_ref(), a8.as_ref());
        return (u, v, alpha)
    }
    else {
        let maxnorm: f64 = 5.371920351148152;
        // Convert T::Real → f64 for the frexp call (f64 is the real type for
        // both f64 and c64 matrices).
        let a_1norm_f64 = a_1norm.to_f64()
            .expect("T::Real must be convertible to f64 for Pade scaling");
        let (_m, _a) = frexp(a_1norm_f64 / maxnorm);
        alpha = _a as isize;
        if alpha < 0 {
            alpha = 0;
        }
        let scale_f64 = (2.0_f64).powi(alpha as i32);
        let a_scaled = a * Scale(from_f64::<T>(1.0 / scale_f64));
        let a2_scaled = a_scaled.as_ref() * a_scaled.as_ref();
        let a4_scaled = a2_scaled.as_ref() * a2_scaled.as_ref();
        let a6_scaled = a4_scaled.as_ref() * a2_scaled.as_ref();
        let (u, v) = pade13(a_scaled.as_ref(), a2_scaled.as_ref(), a4_scaled.as_ref(), a6_scaled.as_ref());
        return (u, v, alpha)
    }
}

// ── Private Padé polynomial helpers ──────────────────────────────────────────
//
// Each function receives pre-computed even powers of A and returns (U, V)
// such that the [p/p] Padé approximant of exp(A) equals (V-U)^{-1}(V+U).

fn pade3<T: ComplexField>(a: MatRef<T>, a2: MatRef<T>) -> (Mat<T>, Mat<T>) {
    const B3: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
    let ident = Mat::<T>::identity(a.ncols(), a.nrows());
    let temp = a2 * Scale(from_f64::<T>(B3[3]))
        + ident.as_ref() * Scale(from_f64::<T>(B3[1]));
    let u = a * temp;
    let v = a2 * Scale(from_f64::<T>(B3[2]))
        + ident.as_ref() * Scale(from_f64::<T>(B3[0]));
    (u, v)
}

fn pade5<T: ComplexField>(a: MatRef<T>, a2: MatRef<T>, a4: MatRef<T>) -> (Mat<T>, Mat<T>) {
    const B5: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
    let ident = Mat::<T>::identity(a.ncols(), a.nrows());
    let temp = a4 * Scale(from_f64::<T>(B5[5]))
        + a2 * Scale(from_f64::<T>(B5[3]))
        + ident.as_ref() * Scale(from_f64::<T>(B5[1]));
    let u = a * temp;
    let v = a4 * Scale(from_f64::<T>(B5[4]))
        + a2 * Scale(from_f64::<T>(B5[2]))
        + ident.as_ref() * Scale(from_f64::<T>(B5[0]));
    (u, v)
}

fn pade7<T: ComplexField>(
    a: MatRef<T>, a2: MatRef<T>, a4: MatRef<T>, a6: MatRef<T>,
) -> (Mat<T>, Mat<T>) {
    const B7: [f64; 8] = [17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.];
    let ident = Mat::<T>::identity(a.ncols(), a.nrows());
    let temp = a6 * Scale(from_f64::<T>(B7[7]))
        + a4 * Scale(from_f64::<T>(B7[5]))
        + a2 * Scale(from_f64::<T>(B7[3]))
        + ident.as_ref() * Scale(from_f64::<T>(B7[1]));
    let u = a * temp;
    let v = a6 * Scale(from_f64::<T>(B7[6]))
        + a4 * Scale(from_f64::<T>(B7[4]))
        + a2 * Scale(from_f64::<T>(B7[2]))
        + ident.as_ref() * Scale(from_f64::<T>(B7[0]));
    (u, v)
}

fn pade9<T: ComplexField>(
    a: MatRef<T>, a2: MatRef<T>, a4: MatRef<T>, a6: MatRef<T>, a8: MatRef<T>,
) -> (Mat<T>, Mat<T>) {
    const B9: [f64; 10] = [
        17643225600., 8821612800., 2075673600.,
        302702400.,   30270240.,   2162160.,
        110880.,      3960.,       90.,
        1.,
    ];
    let ident = Mat::<T>::identity(a.ncols(), a.nrows());
    let temp = a8 * Scale(from_f64::<T>(B9[9]))
        + a6 * Scale(from_f64::<T>(B9[7]))
        + a4 * Scale(from_f64::<T>(B9[5]))
        + a2 * Scale(from_f64::<T>(B9[3]))
        + ident.as_ref() * Scale(from_f64::<T>(B9[1]));
    let u = a * temp;
    let v = a8 * Scale(from_f64::<T>(B9[8]))
        + a6 * Scale(from_f64::<T>(B9[6]))
        + a4 * Scale(from_f64::<T>(B9[4]))
        + a2 * Scale(from_f64::<T>(B9[2]))
        + ident.as_ref() * Scale(from_f64::<T>(B9[0]));
    (u, v)
}

fn pade13<T: ComplexField>(
    a: MatRef<T>, a2: MatRef<T>, a4: MatRef<T>, a6: MatRef<T>,
) -> (Mat<T>, Mat<T>) {
    const B13: [f64; 14] = [
        64764752532480000., 32382376266240000., 7771770303897600.,
        1187353796428800.,  129060195264000.,   10559470521600.,
        670442572800.,      33522128640.,       1323241920.,
        40840800.,          960960.,            16380.,
        182.,               1.,
    ];
    let ident = Mat::<T>::identity(a.ncols(), a.nrows());

    // U polynomial (odd Padé numerator)
    let v1 = a6 * Scale(from_f64::<T>(B13[13]))
        + a4 * Scale(from_f64::<T>(B13[11]))
        + a2 * Scale(from_f64::<T>(B13[9]));
    let temp = a6.as_ref() * v1.as_ref()
        + a6 * Scale(from_f64::<T>(B13[7]))
        + a4 * Scale(from_f64::<T>(B13[5]))
        + a2 * Scale(from_f64::<T>(B13[3]))
        + ident.as_ref() * Scale(from_f64::<T>(B13[1]));
    let u = a * temp;

    // V polynomial (even Padé denominator)
    let temp2 = a6 * Scale(from_f64::<T>(B13[12]))
        + a4 * Scale(from_f64::<T>(B13[10]))
        + a2 * Scale(from_f64::<T>(B13[8]));
    let v2 = a6.as_ref() * temp2.as_ref()
        + a6 * Scale(from_f64::<T>(B13[6]))
        + a4 * Scale(from_f64::<T>(B13[4]))
        + a2 * Scale(from_f64::<T>(B13[2]))
        + ident.as_ref() * Scale(from_f64::<T>(B13[0]));
    (u, v2)
}

/// Computes phi_k(z) for a scalar real or complex `z`.
///
/// Recurrence:
/// ```text
/// phi_0(z) = exp(z)
/// phi_k(z) = (phi_{k-1}(z) - 1/k!) / z
/// ```
pub fn phi_scaler<T: ComplexFloat>(z: T, k: usize) -> T
{
    let mut phi_z = z.exp();
    if k == 0 {
        return phi_z
    }
    for i in 1..=k {
        phi_z = (phi_z - T::from(1.0 / factorial::factorial(i as u64)).unwrap()) / z;
    }
    phi_z
}


#[cfg(test)]
mod test_matexp_pade {
    use std::f64::consts::PI;
    use faer::c64;
    use crate::mat_utils::{random_mat_normal, mat_mat_approx_eq};
    use super::*;

    /// Verify that the recurrence formula and the extension formula for phi_k
    /// agree to within 1e-9 on a random 5×5 real matrix.
    #[test]
    fn test_phi_ext() {
        let dense_a: Mat<f64> = random_mat_normal(5, 5);
        for k in 0..=3 {
            let phi_a     = phi(dense_a.as_ref(), k);
            let phi_ext_a = phi_ext(dense_a.as_ref(), k);
            mat_mat_approx_eq(phi_a.as_ref(), phi_ext_a.as_ref(), 1e-9);
        }
    }

    /// Verify the matrix exponential on a real diagonal matrix.
    ///
    /// For a diagonal matrix A = diag(a1, ..., an), exp(A·dt) is simply
    /// diag(exp(a1·dt), ..., exp(an·dt)).  Off-diagonal entries must remain
    /// zero.  The 1-norm of A·dt ≈ 2.0 routes through the **pade9** branch.
    #[test]
    fn test_matexp_real_diagonal() {
        // A = diag(1.0, 2.0, -1.0),  dt = 1.0
        let diag_vals: [f64; 3] = [1.0, 2.0, -1.0];
        let n = diag_vals.len();
        let dt = 1.0_f64;

        let mut a = Mat::<f64>::zeros(n, n);
        for (i, &v) in diag_vals.iter().enumerate() {
            a[(i, i)] = v;
        }

        let result = matexp(a.as_ref(), dt);
        let tol = 1e-12_f64;

        // Diagonal entries
        for (i, &v) in diag_vals.iter().enumerate() {
            let expected = (v * dt).exp();
            assert!(
                (result[(i, i)] - expected).abs() < tol,
                "result[({i},{i})] expected {expected}, got {}",
                result[(i, i)]
            );
        }

        // Off-diagonal entries must be zero
        for row in 0..n {
            for col in 0..n {
                if row != col {
                    assert!(
                        result[(row, col)].abs() < tol,
                        "result[({row},{col})] expected 0, got {}",
                        result[(row, col)]
                    );
                }
            }
        }
    }

    /// Verify the matrix exponential on a real non-diagonal matrix.
    ///
    /// The skew-symmetric generator of 2-D rotations has the exact result:
    ///
    /// ```text
    /// A  = [[  0, -θ ],       exp(A·dt) = [[ cos(θ·dt), -sin(θ·dt) ],
    ///       [  θ,  0 ]]                    [ sin(θ·dt),  cos(θ·dt) ]]
    /// ```
    ///
    /// With θ = 1.0 and dt = 1.0 the 1-norm of A·dt is 1.0, routing through
    /// the **pade7** branch.  All four entries are non-trivially non-zero.
    #[test]
    fn test_matexp_real_skew_symmetric() {
        let theta = 1.0_f64;
        let dt    = 1.0_f64;

        let a = faer::mat![
            [0.0_f64, -theta],
            [theta,    0.0_f64]
        ];

        let result = matexp(a.as_ref(), dt);

        let tol = 1e-14_f64;
        let (c, s) = ((theta * dt).cos(), (theta * dt).sin());

        assert!(
            (result[(0, 0)] - c).abs() < tol,
            "result[(0,0)]: expected cos({}) = {c}, got {}", theta * dt, result[(0, 0)]
        );
        assert!(
            (result[(0, 1)] - (-s)).abs() < tol,
            "result[(0,1)]: expected -sin({}) = {}, got {}", theta * dt, -s, result[(0, 1)]
        );
        assert!(
            (result[(1, 0)] - s).abs() < tol,
            "result[(1,0)]: expected sin({}) = {s}, got {}", theta * dt, result[(1, 0)]
        );
        assert!(
            (result[(1, 1)] - c).abs() < tol,
            "result[(1,1)]: expected cos({}) = {c}, got {}", theta * dt, result[(1, 1)]
        );
    }

    /// Verify the matrix exponential on a complex diagonal matrix.
    ///
    /// A = diag(iπ, iπ/2)  →  exp(A) = diag(exp(iπ), exp(iπ/2)) = diag(-1, i)
    /// via Euler's identity.
    #[test]
    fn test_matexp_complex_diagonal() {
        let mut a = Mat::<c64>::zeros(2, 2);
        a[(0, 0)] = c64::new(0.0, PI);
        a[(1, 1)] = c64::new(0.0, PI / 2.0);

        let result = matexp(a.as_ref(), 1.0);

        let tol = 1e-12_f64;

        // exp(iπ) = -1 + 0i
        assert!(
            (result[(0, 0)].re + 1.0).abs() < tol,
            "Re(exp(iπ)) expected -1, got {}", result[(0, 0)].re
        );
        assert!(
            result[(0, 0)].im.abs() < tol,
            "Im(exp(iπ)) expected 0, got {}", result[(0, 0)].im
        );

        // exp(iπ/2) = 0 + 1i
        assert!(
            result[(1, 1)].re.abs() < tol,
            "Re(exp(iπ/2)) expected 0, got {}", result[(1, 1)].re
        );
        assert!(
            (result[(1, 1)].im - 1.0).abs() < tol,
            "Im(exp(iπ/2)) expected 1, got {}", result[(1, 1)].im
        );

        // Off-diagonal entries should be zero
        assert!(result[(0, 1)].norm() < tol, "result[(0,1)] expected 0, got {:?}", result[(0, 1)]);
        assert!(result[(1, 0)].norm() < tol, "result[(1,0)] expected 0, got {:?}", result[(1, 0)]);
    }

    /// Verify the matrix exponential on a complex non-diagonal matrix.
    ///
    /// The anti-Hermitian matrix `A = [[0, i], [i, 0]]` shares its
    /// eigenvectors with the real Pauli-X matrix and has eigenvalues `±i`.
    /// The exact result follows from the spectral decomposition
    /// `exp(A) = Q diag(e^i, e^{-i}) Q†`:
    ///
    /// ```text
    /// A  = [[ 0,  i ],       exp(A) = [[ cos(1),   i·sin(1) ],
    ///       [ i,  0 ]]                 [ i·sin(1),  cos(1)   ]]
    /// ```
    ///
    /// With `dt = 1.0` the 1-norm of `A` is 1.0, routing through **pade7**.
    /// All four entries are non-trivially non-zero, and the off-diagonal
    /// entries are purely imaginary — a case that cannot arise in any real
    /// matrix test.
    #[test]
    fn test_matexp_complex_off_diagonal() {
        let mut a = Mat::<c64>::zeros(2, 2);
        a[(0, 1)] = c64::new(0.0, 1.0); // i
        a[(1, 0)] = c64::new(0.0, 1.0); // i

        let result = matexp(a.as_ref(), 1.0);

        let tol = 1e-14_f64;
        let c = 1.0_f64.cos(); // cos(1)
        let s = 1.0_f64.sin(); // sin(1)

        // Diagonal: cos(1) + 0i
        for i in 0..2 {
            assert!(
                (result[(i, i)].re - c).abs() < tol,
                "Re(result[({i},{i})]) expected cos(1) = {c}, got {}", result[(i, i)].re
            );
            assert!(
                result[(i, i)].im.abs() < tol,
                "Im(result[({i},{i})]) expected 0, got {}", result[(i, i)].im
            );
        }

        // Off-diagonal: 0 + i·sin(1), by symmetry of A
        for (r, ci) in [(0, 1), (1, 0)] {
            assert!(
                result[(r, ci)].re.abs() < tol,
                "Re(result[({r},{ci})]) expected 0, got {}", result[(r, ci)].re
            );
            assert!(
                (result[(r, ci)].im - s).abs() < tol,
                "Im(result[({r},{ci})]) expected sin(1) = {s}, got {}", result[(r, ci)].im
            );
        }
    }
}
