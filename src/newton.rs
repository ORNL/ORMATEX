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
//! Jacobian-free Newton-Krylov solvers for implicit time integration.
//!
//! Provides [`jac_newton`], a general Newton solver for a nonlinear residual
//! $G(x) = 0$ with a user supplied linearization, and [`jac_newton_sys`], a
//! solver that works directly on an [`crate::ode_sys::OdeSys`]. In both cases the
//! linear system in each Newton iteration is solved matrix-free with GMRES
//! (`faer_gmres`), so the Jacobian is only accessed through matrix-vector
//! products. Progress is printed to stdout.
//!
//! # References
//!
//! * Knoll, D. A. and Keyes, D. E., "Jacobian-free Newton-Krylov methods: a survey
//!   of approaches and applications", J. Comput. Phys. 193(2) (2004) 357-397,
//!   doi:10.1016/j.jcp.2003.08.010
use crate::ode_sys::*;
use faer::prelude::*;
use faer_gmres::gmres;

/// Newton's method. Solves $G(x) = 0$ for $x$.
///
/// Jacobian-free Newton-Krylov. Iterates
///
/// $$ x_{k+1} = x_k - J^{-1} G(x_k) $$
///
/// where each Newton step is computed by solving $J a = G(x_k)$ for
/// $a = x_k - x_{k+1}$ with GMRES, and then setting $x_{k+1} = x_k - a$.
///
/// The iteration returns $x_k$ when $\Vert G(x_k)\Vert_2 < \mathrm{tol}$ and either
/// the last step was small, $\Vert a\Vert_2 < 0.1 \thinspace (1 + \Vert x_k\Vert_2)$, or at most
/// one Newton step has been taken.
///
/// `'jac` is the lifetime of the Jacobian linear operator (tied to the ODE
/// system object). `x0` is the initial guess and is cloned immediately, so
/// its lifetime is decoupled from `'jac`.
///
/// # Arguments
///
/// * `t` - time passed to `gf` and `gf_jac`
/// * `x0` - initial guess
/// * `gf` - residual function, `gf(t, x)` returns $G(x)$
/// * `gf_jac` - linearization, `gf_jac(t, x)` returns the operator $J$ at $x$
/// * `tol` - convergence tolerance on the 2-norm of the residual $G(x)$
/// * `tol_lin` - tolerance passed to the GMRES linear solver
/// * `iters` - maximum number of Newton iterations
/// * `iters_lin` - maximum number of GMRES iterations per Newton iteration
///
/// # Returns
///
/// The solution $x$ with $G(x) \approx 0$.
///
/// # Errors
///
/// Returns a [`StepError`] with `error_code` 1 if Newton's method does not
/// converge within `iters` iterations.
///
/// # Panics
///
/// Panics if the GMRES linear solve returns an error.
pub fn jac_newton<'jac>(
    t: f64,
    x0: MatRef<'_, f64>,
    gf: &dyn Fn(f64, MatRef<f64>) -> Mat<f64>,
    gf_jac: &dyn Fn(f64, MatRef<f64>) -> ShiftedLinOp<'jac>,
    tol: f64,
    tol_lin: f64,
    iters: usize,
    iters_lin: usize,
) -> Result<Mat<f64>, StepError> {
    println!("=== Newton Solve");
    const TOL_STEP: f64 = 0.1;
    let mut x = x0.to_owned();
    let mut dx_norm = 1.0e20;
    let mut a = faer::Mat::zeros(x.nrows(), x.ncols());
    for i in 0..iters {
        // eval G(x_k)
        let gfn_x = gf(t, x.as_ref());

        // check for g(x) ~= zero
        let norm_f = gfn_x.norm_l2();
        let res_ok = norm_f < tol;
        let step_ok = dx_norm < (TOL_STEP * (1.0 + x.norm_l2()));
        print!("i: {i}, ||f(x_{i})||: {:0.6e} ", norm_f);
        if (step_ok || i <= 1) && res_ok {
            println!("< tol: {:0.5e}. converged after {i} newton steps.", tol);
            return Ok(x);
        }

        let jac_gfn_x = gf_jac(t, x.as_ref());
        // Reset the GMRES solution buffer to zero each iteration so the
        // initial residual is always b (not b - J*a_prev from last iteration).
        a.fill(0.0);
        // solve J * a = G(x_k) for a
        let (lin_err, lin_iters) = gmres(
            jac_gfn_x,
            gfn_x.as_ref(),
            a.as_mut(),
            iters_lin,
            tol_lin,
            None,
        )
        .unwrap();
        // apply a:  x_k+1 = x_k - a
        x = x - a.as_ref();
        dx_norm = a.norm_l2();
        println!(
            ", ||x_{}-x_{}||: {:0.6e},  Lin iters: {lin_iters}, Lin res: {:0.6e}",
            i + 1,
            i,
            dx_norm,
            lin_err
        );
    }
    let err = StepError {
        error_code: 1,
        msg: format!("Newton Failed"),
    };
    Err(err)
}

/// Newton's method applied directly to an ODE system.
///
/// Iterates
///
/// $$ x_{k+1} = x_k - W^{-1} f(t, x_k), \qquad W = \gamma M + s J(t, x_k) $$
///
/// where $f$ is the system rhs, $J$ its Jacobian, $M$ the optional mass matrix
/// and $W$ is obtained from [`crate::ode_sys::OdeSys::fjac_shifted`]. Each step
/// solves $W a = f(t, x_k)$ with GMRES and sets $x_{k+1} = x_k - a$. With
/// $s = 1$ and $\gamma = 0$ this is the standard Newton iteration for
/// $f(t, x) = 0$.
///
/// The iteration returns when the 2-norm of the Newton step, $\Vert a\Vert_2$, is
/// less than `tol`. Note that the GMRES solution buffer is not reset between
/// iterations, so the previous step is used as the initial guess.
///
/// # Arguments
///
/// * `t` - time at which the rhs and Jacobian are evaluated
/// * `scale` - Jacobian scale factor $s$
/// * `gamma` - shift factor $\gamma$
/// * `x0` - initial guess
/// * `sys` - the ODE system
/// * `tol` - convergence tolerance on the 2-norm of the Newton step
/// * `tol_lin` - tolerance passed to the GMRES linear solver
/// * `iters` - maximum number of Newton iterations
/// * `iters_lin` - maximum number of GMRES iterations per Newton iteration
///
/// # Returns
///
/// The converged state $x$.
///
/// # Errors
///
/// Returns a [`StepError`] with `error_code` 1 if Newton's method does not
/// converge within `iters` iterations.
///
/// # Panics
///
/// Panics if the GMRES linear solve returns an error.
pub fn jac_newton_sys<'a>(
    t: f64,
    scale: f64,
    gamma: f64,
    x0: MatRef<f64>,
    sys: &'a dyn OdeSys<'a>,
    tol: f64,
    tol_lin: f64,
    iters: usize,
    iters_lin: usize,
) -> Result<Mat<f64>, StepError> {
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
            &jac_gfn_x,
            gfn_x.as_ref(),
            a.as_mut(),
            iters_lin,
            tol_lin,
            None,
        )
        .unwrap();
        // apply a:  x_k+1 = x_k - a
        x = x.as_ref() - a.as_ref();
        let x_new_norm = a.norm_l2();
        println!("Nonlinear iter: {i}, ||x_{} - x_{}||: {:0.6e},  Linear iters: {lin_iters}, Linear res: {:0.6e}", i+1, i, x_new_norm, lin_err);
        if (x_new_norm) < tol {
            return Ok(x);
        }
    }
    let err = StepError {
        error_code: 1,
        msg: format!("Newton Failed"),
    };
    Err(err)
}

#[cfg(test)]
mod test_newton {
    use crate::test_common::*;
    use assert_approx_eq::assert_approx_eq;

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
        let xsol = jac_newton_sys(
            0.0,
            scale,
            shift,
            x0.as_ref(),
            &my_test_sys,
            tol,
            1e-14,
            100,
            1000,
        )
        .unwrap();

        print!("sol: {:?}", xsol);
        assert_approx_eq!(xsol.get(0, 0), 1.0, tol * 10.);
    }
}
