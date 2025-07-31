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
/// Computes the matrix exponential and matexpv using contour integral appraoch
use faer::prelude::*;
use rayon::prelude::*;
use crate::mat_utils::{real_mat, complex_mat_scale, mat_pow};
use crate::matexp_pade::DensePhikvEvaluator;
use faer::linalg::solvers::{Solve, DenseSolveCore};


#[derive(Debug)]
pub struct CauchyExpm
{
    /// poles
    theta: Mat<c64>,

    /// weights
    alpha: Mat<c64>,

    /// offset
    alpha_0: c64,
}

impl CauchyExpm {
    pub fn new(theta: MatRef<c64>, alpha: MatRef<c64>, alpha_0: c64) -> Self {
        if theta.nrows() != alpha.nrows() {
            panic!("n theta must equal n alpha");
        }
        Self {
            theta: theta.to_owned(),
            alpha: alpha.to_owned(),
            alpha_0: alpha_0
        }
    }

    fn order(&self) -> usize {
        self.theta.nrows()*2
    }

    /// Computes exp(A*dt) for dense A
    pub fn matexp_dense_cauchy(&self, a: MatRef<f64>, dt: f64) -> Mat<f64>
    {
        let s = self.theta.nrows();
        let dim = a.nrows();
        let ident: Mat<c64> = Mat::identity(dim, dim);

        // scaled a and conv. to complex
        let a_dt: Mat<c64> = complex_mat_scale(a, dt);

        // loop over poles
        // let mut exp_a: Mat<c64> = Mat::zeros(dim, dim);
        // for k in 0..s {
        //     let tmp_a = a_dt.as_ref() - Scale(self.theta[(k, 0)])*ident.as_ref();
        //     // pseudo inverse used here as A may be only left invertible
        //     let _tmp_a_qr = tmp_a.qr();
        //     exp_a = exp_a + _tmp_a_qr.inverse() * Scale(self.alpha[(k, 0)]);
        // }
        // same as above in parallel
        let exp_a: Mat<c64> = (0..s).into_par_iter().map(|k| {
                let tmp_a = a_dt.as_ref() - Scale(self.theta[(k, 0)])*ident.as_ref();
                let _tmp_a_qr = tmp_a.qr();
                _tmp_a_qr.inverse() * Scale(self.alpha[(k, 0)])
            }).reduce_with(|a, b| a + b).unwrap();

        // take real components
        let mut rexp_a = 2. * real_mat(exp_a.as_ref());
        // apply shift
        rexp_a = rexp_a + self.alpha_0.re*real_mat(ident.as_ref());
        rexp_a
    }

    /// Extension formula for computing higher order phi functions
    fn _phi_ext(&self, a: MatRef<f64>, bu: MatRef<f64>, dt: f64) -> Mat<f64>
    {
        let n = a.nrows();
        let m = a.ncols();
        // compute extended matrix B
        // build by blocks
        // B = [[a, bu], [0, 0]]
        let mut b_ext = Mat::zeros(a.nrows()*2, a.ncols()*2);
        b_ext.submatrix_mut(0, 0, a.nrows(), a.ncols()).copy_from(a);
        b_ext.submatrix_mut(0, a.ncols(), a.nrows(), a.ncols()).copy_from(bu);

        // compute matexp of b_ext
        let phi_a_ext = self.matexp_dense_cauchy(b_ext.as_ref(), dt);

        // extract phi_k+1
        phi_a_ext.get(0..n, m..).to_owned()
    }

    /// Computes phi_1(A*dt) for dense A using the extension formula
    /// T. Schmelzer and L. Trefethen. Evaluating Matrix Functions for
    /// Exponential Integrators via Caratheodory-Fejer Approximation
    /// and Contour Integrals. Electronic Transactions on Numerical Analysis. v 29. 2007.
    pub fn phik_dense_cauchy(&self, a: MatRef<f64>, dt: f64, k: usize) -> Mat<f64>
    {
        assert!(k <= 1);
        match k {
            0 => { self.matexp_dense_cauchy(a, dt) },
            1 => {
                let tmp_zn = mat_pow(a, k);
                let tmp_znp = mat_pow(a, k-1);
                let phi_k = self._phi_ext(tmp_zn.as_ref(), tmp_znp.as_ref(), dt);
                phi_k
            },
            _ => panic!("Only supports phi_k for k<=1 currently"),
        }
    }

    /// Computes exp(A*dt)*v0 for dense A
    pub fn matexp_dense_apply_cauchy(&self, a: MatRef<f64>, dt: f64, v0: MatRef<f64>) -> Mat<f64>
    {
        let s = self.theta.nrows();
        let dim = a.nrows();
        let ident: Mat<c64> = Mat::identity(dim, dim);

        // cast v0 to complex
        let v0_complex = complex_mat_scale(v0.as_ref(), 1.0);

        // scaled a and conv. to complex
        let a_dt: Mat<c64> = complex_mat_scale(a, dt);

        // loop over poles
        // let mut out_v: Mat<c64> = Mat::zeros(dim, 1);
        // for k in 0..s {
        //     let tmp_a = a_dt.as_ref() - Scale(self.theta[(k, 0)])*ident.as_ref();
        //     let tmp_b = Scale(self.alpha[(k, 0)]) * v0_complex.as_ref();
        //     let _lu_tmp_a = tmp_a.partial_piv_lu();
        //     // solve the dense linear system
        //     out_v = out_v + _lu_tmp_a.solve(tmp_b.as_ref());
        // }
        // same as above in parallel
        let out_v: Mat<c64> = (0..s).into_par_iter().map(|k| {
                let tmp_a = a_dt.as_ref() - Scale(self.theta[(k, 0)])*ident.as_ref();
                let tmp_b = Scale(self.alpha[(k, 0)]) * v0_complex.as_ref();
                let _lu_tmp_a = tmp_a.partial_piv_lu();
                // solve the dense linear system
                _lu_tmp_a.solve(tmp_b.as_ref())
            }).reduce_with(|a, b| a + b).unwrap();

        // take real components
        let mut r_v = 2. * real_mat(out_v.as_ref());
        // apply shift
        r_v = r_v + Scale(self.alpha_0.re)*v0.as_ref();
        r_v
    }

    /// Computes phi_l(A*dt)*v0 for dense A using fixed poles for all phi_l
    /// T. Schmelzer and L. Trefethen. Evaluating Matrix Functions for
    /// Exponential Integrators via Caratheodory-Fejer Approximation
    /// and Contour Integrals. Electronic Transactions on Numerical Analysis. v 29. 2007.
    pub fn phik_dense_apply_cauchy(&self, a: MatRef<f64>, dt: f64, v0: MatRef<f64>, l: usize) -> Mat<f64>
    {
        let s = self.theta.nrows();
        let dim = a.nrows();
        let ident: Mat<c64> = Mat::identity(dim, dim);

        // cast v0 to complex
        let v0_complex = complex_mat_scale(v0.as_ref(), 1.0);

        // scaled a and conv. to complex
        let a_dt: Mat<c64> = complex_mat_scale(a, dt);

        // loop over poles
        // let mut out_v: Mat<c64> = Mat::zeros(dim, 1);
        // for k in 0..s {
        //     let zk = self.theta[(k, 0)].powi(l as i32);
        //     let tmp_a = zk * (a_dt.as_ref() - Scale(self.theta[(k, 0)])*ident.as_ref());
        //     let tmp_b = Scale(self.alpha[(k, 0)]) * v0_complex.as_ref();
        //     let _lu_tmp_a = tmp_a.partial_piv_lu();
        //     // solve the dense linear system
        //     out_v = out_v + _lu_tmp_a.solve(tmp_b.as_ref());
        // }
        // same as above in parallel
        let out_v: Mat<c64> = (0..s).into_par_iter().map(|k| {
                let zk = self.theta[(k, 0)].powi(l as i32);
                let tmp_a = Scale(zk) * (a_dt.as_ref() - Scale(self.theta[(k, 0)])*ident.as_ref());
                let _lu_tmp_a = tmp_a.partial_piv_lu();
                let tmp_b = Scale(self.alpha[(k, 0)]) * v0_complex.as_ref();
                // solve the dense linear system
                _lu_tmp_a.solve(tmp_b.as_ref())
            }).reduce_with(|a, b| a + b).unwrap();

        // take real components
        let mut r_v = 2. * real_mat(out_v.as_ref());
        // apply shift
        r_v = r_v + Scale(self.alpha_0.re)*v0.as_ref();
        r_v
    }
}

impl DensePhikvEvaluator for CauchyExpm {
    fn phik_apply(&self, a: MatRef<f64>, dt: f64, v0: MatRef<f64>, k: usize) -> Mat<f64> {
        self.phik_dense_apply_cauchy(a, dt, v0, k)
    }
}


/// Generate expm and phi evaluator
/// Ref:  Pusa, M. Rational Approximations to the Matrix Exponential in Burnup Calculations.
/// Nuclear Science and Engineering, 169(2), 155–167. https://doi.org/10.13182/NSE10-81
pub fn gen_cram_expm(order: usize) -> CauchyExpm
{
    let mut theta: Mat<c64> = Mat::zeros(order/2, 1);
    let mut alpha: Mat<c64> = Mat::zeros(order/2, 1);
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
        },
        _ => panic!("bad order")
    }

    let alpha_0_cram: c64 = c64::new(2.1248537104952237488e-16, 0.);
    CauchyExpm::new(theta.as_ref(), alpha.as_ref(), alpha_0_cram)
}

/// Generate expm and phi evaluator
pub fn gen_parabolic_expm(order: usize) -> CauchyExpm
{
    assert!(order % 2 == 0);
    let mut theta: Mat<c64> = Mat::zeros(order/2, 1);
    let mut theta_out: Mat<c64> = Mat::zeros(order/2, 1);
    let mut alpha: Mat<c64> = Mat::zeros(order/2, 1);
    let im1: c64 = c64::new(0., 1.);
    let order_im: c64 = c64::new(order as f64, 0.);

    let mut idx: usize = 0;
    for i in (1..order).step_by(2) {
        theta[(idx, 0)] = c64::new(std::f64::consts::PI*(i as f64) / (order as f64), 0.);
        idx += 1;
    }

    for i in 0..theta.nrows() {
        let phi = order_im * (
            c64::new(0.1309, 0.)
            - c64::new(0.1194, 0.)*theta[(i, 0)].powi(2)
            + c64::new(0.25, 0.)*theta[(i, 0)]*im1);
        let phi_prime = order_im * (
            c64::new(-2.0*0.1194, 0.)*theta[(i, 0)]
            + c64::new(0.25, 0.)*im1);
        let a = (im1 / order_im) * (
            phi.exp()*phi_prime);
        alpha[(i, 0)] = a;
        theta_out[(i, 0)] = phi;
    }

    let alpha_0_parabolic: c64 = c64::new(0., 0.);
    CauchyExpm::new(theta_out.as_ref(), alpha.as_ref(), alpha_0_parabolic)
}


#[cfg(test)]
mod test_matexp_cauchy {
    use crate::matexp_pade::{matexp, phi_ext};
    use crate::mat_utils::mat_mat_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_cauchy_matexp() {
        // initialize
        let cram = gen_cram_expm(16);

        let test_a: Mat<f64> = mat![
            [-1.0e00,  0.0e+00,  0.0e+00],
            [ 1.0e00, -1.0e+02,  0.0e+00],
            [ 0.0e00,  1.0e+02, -1.0e-02],
        ];
        let dt = 1.0;
        // compute matexp using pade
        let pade_exp_a = matexp(test_a.as_ref(), dt);

        // compute matexp using cauchy
        let cram_exp_a = cram.matexp_dense_cauchy(test_a.as_ref(), dt);

        // compare
        mat_mat_approx_eq(pade_exp_a.as_ref(), cram_exp_a.as_ref(), 1e-12);
    }

    #[test]
    fn test_cauchy_phi() {
        // initialize
        let cram = gen_cram_expm(16);

        let test_a: Mat<f64> = mat![
            [-1.0e00,  0.0e+00,  0.0e+00],
            [ 1.0e00, -1.0e+02,  0.0e+00],
            [ 0.0e00,  1.0e+02, -1.0e-02],
        ];
        let dt = 1.0;
        // compute phi_k using pade
        let pade_phi1_a = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), 1);

        // compute phi_k using cauchy
        let cram_phi1_a = cram.phik_dense_cauchy(test_a.as_ref(), dt, 1);

        // compare
        mat_mat_approx_eq(pade_phi1_a.as_ref(), cram_phi1_a.as_ref(), 1e-12);

        // higher order phi fns (TODO: fix, broken)
        // let pade_phi2_a = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), 2);
        // let cram_phi2_a = cram.phik_dense_cauchy(test_a.as_ref(), dt, 2);
        // mat_mat_approx_eq(pade_phi2_a.as_ref(), cram_phi2_a.as_ref(), 1e-12);
    }

    #[test]
    fn test_cauchy_cram_phik_apply() {
        // initialize
        let cram = gen_cram_expm(16);

        let test_a: Mat<f64> = mat![
            [-1.0e00,  0.0e+00,  0.0e+00],
            [ 1.0e00, -1.0e+02,  0.0e+00],
            [ 0.0e00,  1.0e+02, -1.0e-02],
        ];
        let dt = 1.0;
        let v0: Mat<f64> = mat![
            [ 1.0e00],
            [ 2.0e00],
            [ 3.0e00],
        ];
        // compute phi_k(a*dt)*v0 using pade
        let pade_phi1_av = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), 1)*v0.as_ref();

        // compute phi_k(a*dt)*v0 using caratheodory-fejer aprroximation and contour integral
        // appraoch
        let cram_phi1_av = cram.phik_dense_apply_cauchy(test_a.as_ref(), dt, v0.as_ref(), 1);
        println!("pade phi1(a*dt)*v0 {:?}", pade_phi1_av.as_ref());
        println!("cram phi1(a*dt)*v0 {:?}", cram_phi1_av.as_ref());
        mat_mat_approx_eq(pade_phi1_av.as_ref(), cram_phi1_av.as_ref(), 1e-12);

        // higher order phi fns
        for k in 0..4 {
            let pade_phik_av = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), k)*v0.as_ref();
            let cram_phik_av = cram.phik_dense_apply_cauchy(test_a.as_ref(), dt, v0.as_ref(), k);
            // we expect some accuracy degredation for higher order phi functions
            mat_mat_approx_eq(pade_phik_av.as_ref(), cram_phik_av.as_ref(), 1e-10);
        }
    }

    #[test]
    fn test_cauchy_parabolic_phik_apply() {
        let parabolic = gen_parabolic_expm(32);
        let test_a: Mat<f64> = mat![
            [-1.0e00,  0.0e+00,  0.0e+00],
            [ 1.0e00, -1.0e+02,  0.0e+00],
            [ 0.0e00,  1.0e+02, -1.0e-02],
        ];
        let dt = 1.0;
        let v0: Mat<f64> = mat![
            [ 1.0e00],
            [ 2.0e00],
            [ 3.0e00],
        ];
        let pade_phi1_av = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), 1)*v0.as_ref();
        let parabolic_phi1_av = parabolic.phik_dense_apply_cauchy(test_a.as_ref(), dt, v0.as_ref(), 1);
        println!("pade phi1(a*dt)*v0 {:?}", pade_phi1_av.as_ref());
        println!("parabolic phi1(a*dt)*v0 {:?}", parabolic_phi1_av.as_ref());
        mat_mat_approx_eq(pade_phi1_av.as_ref(), parabolic_phi1_av.as_ref(), 1e-10);
        for k in 0..4 {
            let pade_phik_av = phi_ext((Scale(dt)*test_a.as_ref()).as_ref(), k)*v0.as_ref();
            let parabolic_phik_av = parabolic.phik_dense_apply_cauchy(test_a.as_ref(), dt, v0.as_ref(), k);
            // we expect some accuracy degredation for higher order phi functions
            mat_mat_approx_eq(pade_phik_av.as_ref(), parabolic_phik_av.as_ref(), 1e-8);
        }
    }
}
