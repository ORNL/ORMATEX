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
//! Contour integral (Cauchy) matrix exponential and phi-function evaluation for dense faer matrices.
//!
//! The matrix exponential and phi-functions are approximated by a quadrature
//! of the Cauchy integral
//!
//! $$ \varphi_k(A) = \frac{1}{2 \pi i} \int_\Gamma \frac{e^z}{z^k} (zI - A)^{-1} \thinspace dz $$
//!
//! which yields a partial fraction (pole) expansion with complex poles
//! $\theta_j$ and weights $\alpha_j$. Only the poles in the upper half plane
//! are stored, and the conjugate pairs are accounted for by taking twice the
//! real part. [`CauchyExpm`] holds the poles and weights and evaluates
//! $\exp(A t)$ and $\varphi_k(A t)$ (as dense matrices or as products with
//! vectors) using one complex linear solve per pole. The poles are processed
//! in parallel with rayon. [`gen_cram_expm`] builds the Chebyshev rational
//! approximation (CRAM) of order 16 and [`gen_parabolic_expm`] builds a
//! quadrature on a parabolic contour.
//!
//! # References
//!
//! * T. Schmelzer, L. N. Trefethen, "Evaluating matrix functions for
//!   exponential integrators via Caratheodory-Fejer approximation and contour
//!   integrals", Electronic Transactions on Numerical Analysis 29 (2007) 1-18.
//! * M. Pusa, "Rational approximations to the matrix exponential in burnup
//!   calculations", Nuclear Science and Engineering 169(2) (2011) 155-167,
//!   doi:10.13182/NSE10-81.
use crate::mat_utils::{complex_mat_scale, real_mat};
use crate::matexp_traits::DensePhikvEvaluator;
use faer::linalg::solvers::{DenseSolveCore, PartialPivLu, Solve};
use faer::prelude::*;
use faer_traits::math_utils::from_f64;
use faer_traits::{ComplexField, RealField};
use num_traits::Float;
use num_complex::Complex;
use rayon::prelude::*;

/// Partial fraction (contour integral) evaluator of the matrix exponential and phi-functions.
///
/// Stores the poles $\theta_j$ and weights $\alpha_j$, $j = 1, \ldots, s$, of a
/// rational approximation, together with the offset $\alpha_0$, so that
///
/// $$ \exp(A) \approx \alpha_0 I + 2 \thinspace \mathrm{Re} \sum_{j=1}^{s} \alpha_j (A - \theta_j I)^{-1} $$
///
/// and, for $\varphi_k$ with $k \ge 1$ applied to a vector $v$,
///
/// $$ \varphi_k(A) v \approx 2 \thinspace \mathrm{Re} \sum_{j=1}^{s} \frac{\alpha_j}{\theta_j^k} (A - \theta_j I)^{-1} v $$
///
/// The stored poles are those in the upper half plane; their conjugates are
/// implied. The total order of the approximation is $2 s$. The weights
/// $\alpha_j$ absorb the quadrature weights, the factor $e^{\theta_j}$ and the
/// sign of the resolvent. Use [`gen_cram_expm`] or [`gen_parabolic_expm`] to
/// construct standard instances.
///
/// The evaluator can cache the LU factorizations of $(A\thinspace dt - \theta_j I)$ via
/// [`DensePhikvEvaluator::apply_prepare`].
#[derive(Debug)]
pub struct CauchyExpm<T> {
    /// poles
    theta: Mat<Complex<T>>,

    /// weights
    alpha: Mat<Complex<T>>,

    /// offset
    alpha_0: Complex<T>,

    /// LU decomp storage
    lu_factors: Option<Vec<PartialPivLu<Complex<T>>>>,
}

impl<T> CauchyExpm<T>
where
    T: RealField + Float + Send + Sync,
    Complex<T>: ComplexField<Real = T>,
{
    /// Creates a contour integral evaluator from poles and weights.
    ///
    /// # Arguments
    ///
    /// * `theta` - column matrix of the $s$ poles $\theta_j$ (upper half plane)
    /// * `alpha` - column matrix of the $s$ weights $\alpha_j$
    /// * `alpha_0` - offset $\alpha_0$ (the limit of the approximation at infinity)
    ///
    /// # Panics
    ///
    /// Panics if `theta` and `alpha` have different numbers of rows.
    pub fn new(theta: MatRef<Complex<T>>, alpha: MatRef<Complex<T>>, alpha_0: Complex<T>) -> Self {
        if theta.nrows() != alpha.nrows() {
            panic!("n theta must equal n alpha");
        }
        Self {
            theta: theta.to_owned(),
            alpha: alpha.to_owned(),
            alpha_0: alpha_0,
            lu_factors: None,
        }
    }

    fn order(&self) -> usize {
        self.theta.nrows() * 2
    }

    /// Computes $\exp(A \thinspace dt)$ for a dense matrix `A`.
    ///
    /// Uses a numerical quadrature scheme to estimate the Cauchy integral
    ///
    /// $$ \exp(A) = \frac{1}{2\pi i} \int_\Gamma e^z (zI - A)^{-1} \thinspace dz $$
    ///
    /// which is approximated by
    ///
    /// $$ \exp(A) \approx \sum_k c_k (z_k I - A)^{-1}, \qquad
    ///    c_k = \frac{w_k e^{z_k}}{2 \pi i} $$
    ///
    /// where $z_k$ are the quadrature points on the contour and $w_k$ the
    /// quadrature weights. In terms of the stored poles and weights this
    /// evaluates $\alpha_0 I + 2 \thinspace \mathrm{Re} \sum_j \alpha_j (A \thinspace dt - \theta_j I)^{-1}$.
    /// A QR factorization is computed for each pole (in parallel), and the
    /// cached LU factors are not used.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - real time step scale factor
    ///
    /// # Returns
    ///
    /// The dense real matrix approximating $\exp(A \thinspace dt)$.
    pub fn matexp_dense_cauchy(&self, a: MatRef<T>, dt: f64) -> Mat<T> {
        let s = self.theta.nrows();
        let dim = a.nrows();
        let ident: Mat<Complex<T>> = Mat::identity(dim, dim);

        // scaled a and conv. to complex
        let a_dt: Mat<Complex<T>> = complex_mat_scale(a, dt);

        // loop over poles in parallel
        let exp_a: Mat<Complex<T>> = (0..s)
            .into_par_iter()
            .map(|k| {
                let tmp_a = a_dt.as_ref() - Scale(self.theta[(k, 0)]) * ident.as_ref();
                let _tmp_a_qr = tmp_a.qr();
                _tmp_a_qr.inverse() * Scale(self.alpha[(k, 0)])
            })
            .reduce_with(|a, b| a + b)
            .unwrap();

        // take real components
        let mut rexp_a: Mat<T> = Scale(from_f64::<T>(2.0)) * real_mat::<T>(exp_a.as_ref());
        // apply shift
        rexp_a = rexp_a + Scale(self.alpha_0.re) * real_mat::<T>(ident.as_ref());
        rexp_a
    }

    // Extension formula for computing higher order phi functions
    fn phik_cauchy_ext(&self, z: MatRef<T>, k: usize) -> Mat<T> {
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
                z_ext.get_mut(0..n, 0..m).copy_from(z);
                z_ext
                    .get_mut(0..z_ext_k_nrows, z_ext_k_ncols..)
                    .copy_from(Mat::<T>::identity(k * n, k * n));
                z_ext
            }
        };
        let phi_ks = self.matexp_dense_cauchy(z_ext.as_ref(), 1.0);
        phi_ks.get(0..n, phi_ks.ncols() - n..).to_owned()
    }

    /// Computes $\varphi_k(A \thinspace dt)$ for a dense matrix `A` using the extension formula.
    ///
    /// For $k = 0$ this is [`CauchyExpm::matexp_dense_cauchy`]. For $k \ge 1$ the
    /// matrix $A \thinspace dt$ is embedded in an $n(k+1) \times n(k+1)$ block matrix
    /// whose exponential contains $\varphi_k(A \thinspace dt)$ in its top-right block, and
    /// the exponential is computed by the contour integral.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - real time step scale factor
    /// * `k` - phi-function order
    ///
    /// # Returns
    ///
    /// The dense real matrix approximating $\varphi_k(A \thinspace dt)$.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not square.
    ///
    /// # References
    ///
    /// * T. Schmelzer, L. N. Trefethen, "Evaluating matrix functions for
    ///   exponential integrators via Caratheodory-Fejer approximation and
    ///   contour integrals", Electronic Transactions on Numerical Analysis 29
    ///   (2007) 1-18.
    pub fn phik_dense_cauchy(&self, a: MatRef<T>, dt: f64, k: usize) -> Mat<T> {
        match k {
            0 => self.matexp_dense_cauchy(a, dt),
            _ => {
                self.phik_cauchy_ext((Scale(from_f64::<T>(dt)) * a).as_ref(), k)
            }
        }
    }

    /// Computes $\exp(A \thinspace dt) v_0$ for a dense matrix `A`.
    ///
    /// Alias to [`CauchyExpm::phik_dense_apply_cauchy`] with $k = 0$.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - real time step scale factor
    /// * `v0` - the vector (single column) to which the matrix exponential is applied
    ///
    /// # Returns
    ///
    /// The vector $\exp(A \thinspace dt) v_0$.
    ///
    /// # Panics
    ///
    /// Panics if `v0` does not have exactly one column.
    pub fn matexp_dense_apply_cauchy(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>) -> Mat<T> {
        self.phik_dense_apply_cauchy(a, dt, v0, vec![0])
    }

    /// Computes a linear combination of phi-function vector products.
    ///
    /// Evaluates
    ///
    /// $$ \sum_j \varphi_{k_j}(dt \thinspace A) \thinspace v_j $$
    ///
    /// where $k_j$ is `ks[j]` and $v_j$ is column `j` of `vb`. For each pole a
    /// single linear solve with multiple right hand sides is performed, and the
    /// columns are combined with the coefficients $2 \alpha_i / \theta_i^{k_j}$.
    /// The offset $\alpha_0$ is applied only to the columns with $k_j = 0$.
    ///
    /// If [`DensePhikvEvaluator::apply_prepare`] has been called, the cached LU
    /// factors are used and `a` and `dt` are ignored. The caller must
    /// call `apply_prepare` again whenever `a` or `dt` change.
    ///
    /// # Arguments
    ///
    /// * `a` - dense square matrix $A$
    /// * `dt` - real time step scale factor
    /// * `vb` - matrix whose columns are the vectors $v_j$
    /// * `ks` - the phi-function orders $k_j$, one per column of `vb`
    ///
    /// # Returns
    ///
    /// A single column vector holding the summed product.
    ///
    /// # Panics
    ///
    /// Panics if the number of columns of `vb` differs from `ks.len()`.
    ///
    /// # References
    ///
    /// * T. Schmelzer, L. N. Trefethen, "Evaluating matrix functions for
    ///   exponential integrators via Caratheodory-Fejer approximation and
    ///   contour integrals", Electronic Transactions on Numerical Analysis 29
    ///   (2007) 1-18.
    pub fn phik_dense_apply_cauchy(
        &self,
        a: MatRef<T>,
        dt: f64,
        vb: MatRef<T>,
        ks: Vec<usize>,
    ) -> Mat<T> {
        assert!(vb.ncols() == ks.len());
        let s = self.theta.nrows();
        let dim = a.nrows();
        let ident: Mat<Complex<T>> = Mat::identity(dim, dim);

        // cast vb to complex
        let vb_complex = complex_mat_scale(vb.as_ref(), 1.0);

        // scaled a and conv. to complex
        let a_dt: Mat<Complex<T>> = complex_mat_scale(a, dt);

        // loop over poles in parallel
        let out_v: Mat<Complex<T>> = (0..s)
            .into_par_iter()
            .map(|i| {
                let zk = ks
                    .iter()
                    .map(|k| from_f64::<Complex<T>>(2.0) * self.alpha[(i, 0)] / self.theta[(i, 0)].powi(*k as i32))
                    .collect::<Vec<Complex<T>>>();
                let coeffs_i = faer::ColRef::from_slice(&zk).as_mat();
                // Solve the linear system
                let solved = match &self.lu_factors {
                    Some(lu_factors) => {
                        // re-use LU decomp if available
                        lu_factors[i].solve(vb_complex.as_ref())
                    }
                    _ => {
                        let _tmp_a = a_dt.as_ref() - Scale(self.theta[(i, 0)]) * ident.as_ref();
                        let lu = _tmp_a.partial_piv_lu();
                        // solve with multiple RHS
                        lu.solve(vb_complex.as_ref())
                    }
                };

                // Apply the PFD coefficients and sum the RHS columns.
                solved * coeffs_i
            })
            .reduce_with(|a, b| a + b)
            .unwrap();

        // take real components
        let mut r_v: Mat<T> = real_mat(out_v.as_ref());

        // If phi_0 was requested, apply shift
        for (idx, &k) in ks.iter().enumerate() {
            // only apply shift to phi_0 col
            if k == 0 {
                let mut r_col = r_v.col_mut(idx);
                r_col += Scale(self.alpha_0.re) * vb.col(idx);
            }
        }

        r_v
    }
}

impl<T> DensePhikvEvaluator<T> for CauchyExpm<T>
where
    T: RealField + Float + Send + Sync,
    Complex<T>: ComplexField<Real = T>,
{
    fn apply_phi_k(&self, a: MatRef<T>, dt: f64, v0: MatRef<T>, k: usize) -> Mat<T> {
        self.phik_dense_apply_cauchy(a, dt, v0, vec![k])
    }

    fn apply_phi_k_v(&self, a: MatRef<T>, dt: f64, vb: &Vec<MatRef<T>>, ks: &Vec<usize>) -> Mat<T>
    {
        assert!(!vb.is_empty());
        assert!(vb.len() == ks.len());
        let mut vb_mat = Mat::zeros(a.nrows(), vb.len());
        for (j, v) in vb.iter().enumerate() {
            assert!(v.nrows() == a.nrows());
            assert!(v.ncols() == 1);
            vb_mat.col_mut(j).copy_from(v.col(0));
        }
        self.phik_dense_apply_cauchy(a, dt, vb_mat.as_ref(), ks.clone())
    }

    fn apply_prepare(&mut self, a: MatRef<T>, dt: f64, _v0: MatRef<T>, _k: usize) {
        let s = self.theta.nrows();
        let dim = a.nrows();
        let ident: Mat<Complex<T>> = Mat::identity(dim, dim);

        // scaled a and conv. to complex
        let a_dt: Mat<Complex<T>> = complex_mat_scale(a, dt);

        // loop over poles in parallel, compute LU decompositions
        let out_lu: Vec<PartialPivLu<Complex<T>>> = (0..s)
            .into_par_iter()
            .map(|i| {
                let tmp_a = a_dt.as_ref() - Scale(self.theta[(i, 0)]) * ident.as_ref();
                tmp_a.partial_piv_lu()
            })
            .collect();

        self.lu_factors = Some(out_lu)
    }
}

/// Cast complex f64 to complex T
fn cast_c1<T: Float>(z: c64) -> Complex<T> {
    Complex::new(T::from(z.re).unwrap(), T::from(z.im).unwrap())
}

fn cast_c<T: Float>(m: MatRef<c64>) -> Mat<Complex<T>> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| cast_c1(m[(i, j)]))
}

/// Generates a CRAM matrix exponential and phi-function evaluator.
///
/// Builds a [`CauchyExpm`] from the poles and weights of the Chebyshev
/// rational approximation method (CRAM) to the exponential on the negative
/// real axis. Only order 16 is currently available, which uses 8 poles in the
/// upper half plane. The approximation is best suited to matrices with
/// spectrum near the negative real axis.
///
/// # Arguments
///
/// * `order` - order of the CRAM approximation; must be 16
///
/// # Returns
///
/// A [`CauchyExpm`] evaluator with the CRAM poles, weights and offset.
///
/// # Panics
///
/// Panics if `order` is not 16.
///
/// # References
///
/// * M. Pusa, "Rational approximations to the matrix exponential in burnup
///   calculations", Nuclear Science and Engineering 169(2) (2011) 155-167,
///   doi:10.13182/NSE10-81.
pub fn gen_cram_expm<T>(order: usize) -> CauchyExpm<T>
where
    T: RealField + Float + Send + Sync,
    Complex<T>: ComplexField<Real = T>,
{
    let mut theta: Mat<c64> = Mat::zeros(order / 2, 1);
    let mut alpha: Mat<c64> = Mat::zeros(order / 2, 1);
    match order {
        16 => {
            // Defines the complex values for CRAM of order 16
            theta[(0, 0)] = c64::new(-10.843917078696988026, 19.277446167181652284);
            theta[(1, 0)] = c64::new(-5.2649713434426468895, 16.220221473167927305);
            theta[(2, 0)] = c64::new(5.9481522689511774808, 3.5874573620183222829);
            theta[(3, 0)] = c64::new(3.5091036084149180974, 8.4361989858843750826);
            theta[(4, 0)] = c64::new(6.4161776990994341923, 1.1941223933701386874);
            theta[(5, 0)] = c64::new(1.4193758971856659786, 10.925363484496722585);
            theta[(6, 0)] = c64::new(4.9931747377179963991, 5.9968817136039422260);
            theta[(7, 0)] = c64::new(-1.4139284624888862114, 13.497725698892745389);

            alpha[(0, 0)] = c64::new(-0.0000005090152186522491565, -0.00002422001765285228797);
            alpha[(1, 0)] = c64::new(0.00021151742182466030907, 0.0043892969647380673918);
            alpha[(2, 0)] = c64::new(113.39775178483930527, 101.9472170421585645);
            alpha[(3, 0)] = c64::new(15.059585270023467528, -5.7514052776421819979);
            alpha[(4, 0)] = c64::new(-64.500878025539646595, -224.59440762652096056);
            alpha[(5, 0)] = c64::new(-1.4793007113557999718, 1.7686588323782937906);
            alpha[(6, 0)] = c64::new(-62.518392463207918892, -11.19039109428322848);
            alpha[(7, 0)] = c64::new(0.041023136835410021273, -0.15743466173455468191);
        }
        _ => panic!("bad order"),
    }

    let alpha_0_cram: c64 = c64::new(2.1248537104952237488e-16, 0.);
    CauchyExpm::new(cast_c(theta.as_ref()).as_ref(), cast_c(alpha.as_ref()).as_ref(), cast_c1(alpha_0_cram))
}

/// Generates a contour integral evaluator on a parabolic contour.
///
/// Builds a [`CauchyExpm`] whose poles are the points
///
/// $$ z(\vartheta) = N \left( 0.1309 - 0.1194 \vartheta^2 + 0.25 i \vartheta \right) $$
///
/// evaluated at $\vartheta_j = \pi (2j - 1) / N$ for $j = 1, \ldots, N/2$, where
/// $N$ is `order`. The weights are $\alpha_j = \frac{i}{N} e^{z_j} z^\prime (\vartheta_j)$
/// (midpoint rule quadrature) and the offset is zero. Only the $N/2$ points in
/// the upper half plane are stored.
///
/// # Arguments
///
/// * `order` - number of quadrature points $N$; must be even
///
/// # Returns
///
/// A [`CauchyExpm`] evaluator with $N/2$ poles.
///
/// # Panics
///
/// Panics if `order` is odd.
pub fn gen_parabolic_expm<T>(order: usize) -> CauchyExpm<T>
where
    T: RealField + Float + Send + Sync,
    Complex<T>: ComplexField<Real = T>,
{
    assert!(order % 2 == 0);
    let mut theta: Mat<c64> = Mat::zeros(order / 2, 1);
    let mut theta_out: Mat<c64> = Mat::zeros(order / 2, 1);
    let mut alpha: Mat<c64> = Mat::zeros(order / 2, 1);
    let im1: c64 = c64::new(0., 1.);
    let order_im: c64 = c64::new(order as f64, 0.);

    let mut idx: usize = 0;
    for i in (1..order).step_by(2) {
        theta[(idx, 0)] = c64::new(std::f64::consts::PI * (i as f64) / (order as f64), 0.);
        idx += 1;
    }

    for i in 0..theta.nrows() {
        let phi = order_im
            * (c64::new(0.1309, 0.) - c64::new(0.1194, 0.) * theta[(i, 0)].powi(2)
                + c64::new(0.25, 0.) * theta[(i, 0)] * im1);
        let phi_prime =
            order_im * (c64::new(-2.0 * 0.1194, 0.) * theta[(i, 0)] + c64::new(0.25, 0.) * im1);
        let a = (im1 / order_im) * (phi.exp() * phi_prime);
        alpha[(i, 0)] = a;
        theta_out[(i, 0)] = phi;
    }

    let alpha_0_parabolic: c64 = c64::new(0., 0.);
    CauchyExpm::new(cast_c(theta_out.as_ref()).as_ref(), cast_c(alpha.as_ref()).as_ref(), cast_c1(alpha_0_parabolic))
}

#[cfg(test)]
mod test_matexp_cauchy {
    use crate::mat_utils::mat_mat_approx_eq;
    use crate::matexp_pade::{matexp, phi_ext};

    // bring everything from above (parent) module into scope
    use super::*;

    fn _gen_test_a() -> Mat<f64> {
        let test_a: Mat<f64> = mat![
            [-1.0e00, 0.0e+00, 0.0e+00],
            [1.0e00, -1.0e+02, 0.0e+00],
            [0.0e00, 1.0e+02, -1.0e-02],
        ];
        test_a
    }

    #[test]
    fn test_cauchy_matexp() {
        // initialize
        let cram = gen_cram_expm(16);
        let test_a = _gen_test_a();
        let dt = 1.0;
        // compute matexp using pade
        let pade_exp_a = matexp(test_a.as_ref(), dt);

        // compute matexp using cauchy
        let cram_exp_a = cram.matexp_dense_cauchy(test_a.as_ref(), dt);

        // compare
        mat_mat_approx_eq(pade_exp_a.as_ref(), cram_exp_a.as_ref(), 1e-12);
    }

    #[test]
    fn test_cauchy_f32() {
        let cram = gen_cram_expm(16);
        let a32: Mat<f32> = Mat::from_fn(3, 3, |i, j| _gen_test_a()[(i, j)] as f32);
        let v: Mat<f32> = mat![[1.0f32], [2.0], [3.0]];
        let out = cram.phik_dense_apply_cauchy(a32.as_ref(), 1.0, v.as_ref(), vec![1]);
        let expected = phi_ext(_gen_test_a().as_ref(), 1) * v.as_ref().to_owned().map(|x| *x as f64).as_ref();
        for i in 0..3 {
            assert!((out[(i, 0)] as f64 - expected[(i, 0)]).abs() < 1e-4);
        }
        let e = cram.matexp_dense_cauchy(a32.as_ref(), 1.0);
        assert_eq!(e.nrows(), 3);
    }

    #[test]
    fn test_cauchy_phi() {
        // initialize
        let cram = gen_cram_expm(16);
        let test_a = _gen_test_a();
        let dt = 1.0;
        // compute phi_k using pade
        let pade_phi1_a = phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), 1);

        // compute phi_k using cauchy
        let cram_phi1_a = cram.phik_dense_cauchy(test_a.as_ref(), dt, 1);

        // compare
        mat_mat_approx_eq(pade_phi1_a.as_ref(), cram_phi1_a.as_ref(), 1e-12);

        // higher order phi fns
        let pade_phi2_a = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), 2);
        let cram_phi2_a = cram.phik_dense_cauchy(test_a.as_ref(), dt, 2);
        mat_mat_approx_eq(pade_phi2_a.as_ref(), cram_phi2_a.as_ref(), 1e-10);
    }

    #[test]
    fn test_cauchy_cram_phik_apply() {
        // initialize
        let mut cram = gen_cram_expm(16);
        let test_a = _gen_test_a();
        let dt = 1.0;
        let v0: Mat<f64> = mat![[1.0e00], [2.0e00], [3.0e00],];
        // compute phi_k(a*dt)*v0 using pade
        let pade_phi1_av = phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), 1) * v0.as_ref();

        // compute phi_k(a*dt)*v0 using caratheodory-fejer approximation
        let cram_phi1_av = cram.apply_phi_k(test_a.as_ref(), dt, v0.as_ref(), 1);
        println!("pade phi1(a*dt)*v0 {:?}", pade_phi1_av.as_ref());
        println!("cram phi1(a*dt)*v0 {:?}", cram_phi1_av.as_ref());
        mat_mat_approx_eq(pade_phi1_av.as_ref(), cram_phi1_av.as_ref(), 1e-12);

        // higher order phi fns
        cram.apply_prepare(test_a.as_ref(), dt, v0.as_ref(), 0);
        for k in 0..4 {
            let pade_phik_av = phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), k) * v0.as_ref();
            let cram_phik_av = cram.apply_phi_k(test_a.as_ref(), dt, v0.as_ref(), k);
            // we expect some accuracy degradation for higher order phi functions
            mat_mat_approx_eq(pade_phik_av.as_ref(), cram_phik_av.as_ref(), 1e-10);
        }
    }

    #[test]
    fn test_cauchy_cram_phik_v_apply() {
        // Test the ability evaluate linear combinations of phi-function-vector prods
        // of the form [phi_0(dt*A) * v0 + phi_1(dt*A) * v1 + ... phi_k(dt*A) * vk]
        let mut cram = gen_cram_expm(16);
        let test_a = _gen_test_a();
        let dt = 1.0;
        let v0: Mat<f64> = mat![[0.0e00], [0.0e00], [0.0e00],];
        let v1: Mat<f64> = mat![[1.1e00], [2.1e00], [3.1e00],];
        let v2: Mat<f64> = mat![[1.2e00], [2.2e00], [3.2e00],];
        let vb = vec![v0.as_ref(), v1.as_ref(), v2.as_ref()];

        // compute ground truth phi-vector products
        let mut expected: Mat<f64> = Mat::zeros(v0.nrows(), v0.ncols());
        for k in 0..=2 {
            expected += phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), k) * vb[k];
        }

        // compute using multi RHS
        cram.apply_prepare(test_a.as_ref(), dt, v0.as_ref(), 0);
        assert!(cram.lu_factors.is_some());
        let out = cram.apply_phi_k_v(test_a.as_ref(), dt, &vb, &vec![0,1,2]);

        // ensure result is near expected within tol
        mat_mat_approx_eq(expected.as_ref(), out.as_ref(), 1e-10);
    }

    #[test]
    fn test_cauchy_cram_phi0_v_apply() {
        // Test [phi_0(dt*A) * v0]
        let mut cram = gen_cram_expm(16);
        let test_a = _gen_test_a();
        let dt = 1.0;
        let v0: Mat<f64> = mat![[1.0e00], [2.0e-2], [3.0e01],];
        let vb = vec![v0.as_ref()];

        // compute ground truth phi-vector products
        let mut expected: Mat<f64> = Mat::zeros(v0.nrows(), v0.ncols());
        for k in 0..vb.len() {
            expected += phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), k) * vb[k];
        }

        // compute using multi RHS
        cram.apply_prepare(test_a.as_ref(), dt, v0.as_ref(), 0);
        assert!(cram.lu_factors.is_some());
        let out = cram.apply_phi_k_v(test_a.as_ref(), dt, &vb, &vec![0,]);

        // ensure result is near expected within tol
        mat_mat_approx_eq(expected.as_ref(), out.as_ref(), 1e-12);
    }

    #[test]
    fn test_cauchy_parabolic_phik_apply() {
        let parabolic = gen_parabolic_expm(32);
        let test_a = _gen_test_a();
        let dt = 1.0;
        let v0: Mat<f64> = mat![[1.0e00], [2.0e00], [3.0e00],];
        let pade_phi1_av = phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), 1) * v0.as_ref();
        let parabolic_phi1_av = parabolic.apply_phi_k(test_a.as_ref(), dt, v0.as_ref(), 1);
        println!("pade phi1(a*dt)*v0 {:?}", pade_phi1_av.as_ref());
        println!("parabolic phi1(a*dt)*v0 {:?}", parabolic_phi1_av.as_ref());
        mat_mat_approx_eq(pade_phi1_av.as_ref(), parabolic_phi1_av.as_ref(), 1e-10);
        for k in 0..4 {
            let pade_phik_av = phi_ext((Scale(dt) * test_a.as_ref()).as_ref(), k) * v0.as_ref();
            let parabolic_phik_av = parabolic.apply_phi_k(test_a.as_ref(), dt, v0.as_ref(), k);
            // we expect some accuracy degradation for higher order phi functions
            mat_mat_approx_eq(pade_phik_av.as_ref(), parabolic_phik_av.as_ref(), 1e-8);
        }
    }
}
