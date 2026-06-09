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
/// Newtons methods for implicit methods
use faer::prelude::*;
use crate::ode_sys::*;
use faer_gmres::gmres;


/// Newton's method. Solves G(x)=0 for x.
/// Jacobian-free newton krylov
///
/// Iterates x_k+1 = x_k - J^-1 * G(x_k)
/// or
/// G(x_k) = J * (x_k - x_k+1) = J * a
/// solve J * a = G(x_k) for a. with a = x_k - x_k+1 then
/// x_k+1 = x_k - a
///
/// `'jac` is the lifetime of the Jacobian linear operator (tied to the ODE
/// system object). `x0` is the initial guess and is cloned immediately, so
/// its lifetime is decoupled from `'jac`.
///
pub fn jac_newton <'jac> (
        t: f64,
        x0: MatRef<'_, f64>,
        gf: &dyn Fn(f64, MatRef<f64>) -> Mat<f64>,
        gf_jac: &dyn Fn(f64, MatRef<f64>) -> ShiftedLinOp<'jac>,
        tol: f64,
        tol_lin: f64,
        iters: usize,
        iters_lin: usize,
        ) -> Result<Mat<f64>, StepError>
    {
    println!("=== Newton Solve");
    let mut x = x0.to_owned();
    let mut a = faer::Mat::zeros(x.nrows(), x.ncols());
    for i in 0..iters {
        // eval G(x_k)
        let gfn_x = gf(t, x.as_ref());
        let jac_gfn_x = gf_jac(t, x.as_ref());
        // Reset the GMRES solution buffer to zero each iteration so the
        // initial residual is always b (not b - J*a_prev from last iteration).
        a.fill(0.0);
        // solve J * a = G(x_k) for a
        let (lin_err, lin_iters) = gmres(
            jac_gfn_x, gfn_x.as_ref(), a.as_mut(), iters_lin, tol_lin, None).unwrap();
        // apply a:  x_k+1 = x_k - a
        x = x - a.as_ref();
        let x_new_norm = a.norm_l2();
        println!("Nonlinear iter: {i}, ||x_{} - x_{}||: {:0.6e},  Linear iters: {lin_iters}, Linear res: {:0.6e}", i+1, i, x_new_norm, lin_err);
        if (x_new_norm) < tol {
            return Ok(x);
        }
    }
    let err = StepError{error_code: 1, msg: format!("Newton Failed")};
    Err(err)
}


/// Newton's method. Solves G(x)=0 for x.
/// Iterates x_k+1 = x_k - J^-1 * G(x_k)
///
pub fn jac_newton_sys <'a> (
        t: f64,
        scale: f64,
        gamma: f64,
        x0: MatRef<f64>,
        sys: &'a dyn OdeSys<'a>,
        tol: f64,
        tol_lin: f64,
        iters: usize,
        iters_lin: usize,
        ) -> Result<Mat<f64>, StepError>
    {
    println!("=== Newton Solve");
    let mut x: Mat<f64> = x0.to_owned();
    let mut a = faer::Mat::zeros(x.nrows(), x.ncols());
    for i in 0..iters {
        // eval G(x_k)
        let gfn_x = sys.frhs(t, x.as_ref());
        // let jac_gfn_x = sys.fjac_shifted(t, x.as_ref(), 1.0, None);
        // let lin_x = x.clone();
        let jac_gfn_x = sys.fjac_shifted(t, x.as_ref(), scale, Some(gamma));
        // solve J * a = G(x_k) for a
        let (lin_err, lin_iters) = gmres(
            &jac_gfn_x, gfn_x.as_ref(), a.as_mut(), iters_lin, tol_lin, None).unwrap();
        // apply a:  x_k+1 = x_k - a
        x = x.as_ref() - a.as_ref();
        let x_new_norm = a.norm_l2();
        println!("Nonlinear iter: {i}, ||x_{} - x_{}||: {:0.6e},  Linear iters: {lin_iters}, Linear res: {:0.6e}", i+1, i, x_new_norm, lin_err);
        if (x_new_norm) < tol {
            return Ok(x);
        }
    }
    let err = StepError{error_code: 1, msg: format!("Newton Failed")};
    Err(err)
}


#[cfg(test)]
mod test_newton {
    use assert_approx_eq::assert_approx_eq;
    use crate::test_common::*;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_newton_quad() {
        let init_sys_x: Mat<f64> = faer::Mat::full(1, 1, 2.0);
        let my_test_sys = TestQuadSys::new(init_sys_x);

        let x0: Mat<f64> = faer::Mat::full(1, 1, 2.0);

        // Solve the nonlinear sys
        let scale = 1.0;
        let shift = 0.0;
        let tol = 1e-8;
        let xsol = jac_newton_sys(0.0, scale, shift, x0.as_ref(), &my_test_sys, tol, 1e-14, 100, 1000).unwrap();

        print!("sol: {:?}", xsol);
        assert_approx_eq!(xsol.get(0, 0), 1.0, tol*10.);

    }

}
