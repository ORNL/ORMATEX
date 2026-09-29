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

//! Implicit time integrators.
//!
//! Provides implicit integrators for $y^\prime  = f(t, y)$ that solve a nonlinear system
//! in each step with a Jacobian-free Newton-Krylov method (see [`crate::newton`]):
//!
//! * [`DirkIntegrator`] - generic DIRK / SDIRK / ESDIRK methods defined by an
//!   [`ImplicitBT`] Butcher tableau (implicit Euler, Crank-Nicolson, SDIRK).
//! * [`BdfIntegrator`] - the BDF1 (backward Euler) and BDF2 linear multistep
//!   methods with constant step size.
//!
//! Both integrators implement [`crate::ode_traits::IntegrateSys`], use fixed step
//! sizes and provide no embedded error estimate.
//!
//! # Mass matrices
//!
//! An [`OdeSys`] may supply a mass matrix $M$ through [`OdeSys::fmass`], which
//! describes the problem $M y^\prime = f(t, y)$. Both integrators support this,
//! including a singular $M$ (index one differential-algebraic systems) in the
//! cases noted below. Without a mass matrix ($M = I$) the arithmetic is the
//! same as for a plain ODE.
//!
//! For a DIRK method the stage derivatives are $k_i = M^{-1} f(t_i, Y_i)$ and
//! $Y_i = y_n + \Delta t \sum_j a_{ij} k_j$. Each implicit stage solves
//!
//! $$ G(Y_i) = M (Y_i - Y_{expl}) - \Delta t \thinspace a_{ii} f(t_i, Y_i) = 0 $$
//!
//! with Newton's method, where $Y_{expl} = y_n + \Delta t \sum_{j<i} a_{ij} k_j$.
//! The Newton matrix is exactly the Jacobian of $G$,
//!
//! $$ W = M - \Delta t \thinspace a_{ii} J, $$
//!
//! which is provided by [`OdeSys::fjac_shifted`] with $\gamma = 1$ and
//! $s = -\Delta t \thinspace a_{ii}$. The stage derivative of an implicit stage is
//! recovered as $k_i = (Y_i - Y_{expl}) / (\Delta t \thinspace a_{ii})$, so no solve with
//! $M$ is needed. BDF2 solves
//! $M (y - \frac{4}{3} y_n + \frac{1}{3} y_{n-1}) - \frac{2}{3} \Delta t \thinspace f(t_{n+1}, y) = 0$
//! with $W = M - \frac{2}{3} \Delta t J$, and BDF1 is the implicit Euler DIRK method.
//!
//! Notes and limitations:
//!
//! * An explicit stage ($a_{ii} = 0$, for example the first stage of
//!   Crank-Nicolson) needs $k = M^{-1} f$, which is computed with GMRES (using
//!   the linear tolerance and iteration limit of the integrator). Such methods
//!   therefore require a nonsingular $M$. BDF1, BDF2 and the SDIRK methods have
//!   only implicit stages and work with singular $M$.
//! * For a time dependent $M(t)$, $M$ is evaluated at the stage time $t_i$
//!   (at $t_{n+1}$ for BDF2). This is an approximation, exact for a constant $M$.
//! * For a differential-algebraic system the initial state should satisfy the
//!   algebraic constraints. Order reduction may occur for the algebraic
//!   components.
//!
//! # References
//!
//! * Alexander, R., "Diagonally implicit Runge-Kutta methods for stiff O.D.E.'s",
//!   SIAM J. Numer. Anal. 14(6) (1977) 1006-1021
//! * Hairer, E. and Wanner, G., "Solving Ordinary Differential Equations II:
//!   Stiff and Differential-Algebraic Problems", 2nd ed., Springer Series in
//!   Computational Mathematics 14, Springer, 1996
use crate::newton::*;
use crate::ode_sys::*;
use crate::ode_traits::IntegrateSys;
use crate::tableau_implicit::ImplicitBT;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer_gmres::gmres;
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Compute the product $M v$ of a mass matrix operator with a matrix `v`.
fn apply_mass(mass: &dyn LinOp<f64>, v: MatRef<'_, f64>) -> Mat<f64> {
    let par = faer::get_global_parallelism();
    let mut out = Mat::zeros(mass.nrows(), v.ncols());
    let mut buf = MemBuffer::new(mass.apply_scratch(v.ncols(), par));
    mass.apply(out.as_mut(), v, par, MemStack::new(&mut buf));
    out
}

/// Sized wrapper of a `&dyn LinOp`, so that it can be passed to GMRES.
struct MassRef<'m>(&'m dyn LinOp<f64>);

impl<'m> std::fmt::Debug for MassRef<'m> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MassRef")
    }
}

impl<'m> LinOp<f64> for MassRef<'m> {
    fn apply_scratch(&self, rhs_ncols: usize, par: faer::Par) -> faer::dyn_stack::StackReq {
        self.0.apply_scratch(rhs_ncols, par)
    }
    fn nrows(&self) -> usize {
        self.0.nrows()
    }
    fn ncols(&self) -> usize {
        self.0.ncols()
    }
    fn apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: faer::Par, stack: &mut MemStack) {
        self.0.apply(out, rhs, par, stack)
    }
    fn conj_apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: faer::Par, stack: &mut MemStack) {
        self.0.conj_apply(out, rhs, par, stack)
    }
}

/// Solve $M k = f$ for $k$ with GMRES. Used to obtain the stage derivative
/// $k = M^{-1} f$ of an explicit stage when a mass matrix is present.
fn solve_mass(
    mass: &dyn LinOp<f64>,
    f: MatRef<'_, f64>,
    tol_lin: f64,
    iters_lin: usize,
) -> Result<Mat<f64>, StepError> {
    let mut k = Mat::zeros(f.nrows(), f.ncols());
    gmres(MassRef(mass), f, k.as_mut(), iters_lin, tol_lin, None).map_err(|_| StepError {
        error_code: 2,
        msg: String::from("mass matrix solve (GMRES) failed"),
    })?;
    Ok(k)
}

/// Advance `y' = f(t,y)` by one step `dt` using the implicit Butcher tableau `bt`.
///
/// For each stage `i`:
///
/// 1. Build the explicit accumulation
///    `y_expl = y0 + dt * \sum_{j < i} a[i][j] * k[j]`
///
/// 2. Explicit stage. (`a[i][i] == 0`):
///    `k[i] = f(t + c[i]*dt, y_expl)`
///
/// 3. Implicit stage. (`a[i][i] != 0`):
///    Solve  `g(y_i) = y_i - y_expl - dt*a[i][i]*f(t_i, y_i) = 0`
///    using Newton-Krylov, starting from `y_expl`, with Jacobian
///    `I - dt*a[i][i]*J_f == fjac_shifted(t_i, y_i, -dt*a[i][i], gamma=1)`.
///    Then `k[i] = f(t_i, y_i)`.
///
/// Finally `y_{n+1} = y0 + dt * \sum_i b[i] * k[i]`.
///
/// # Mass matrix
///
/// With a mass matrix $M$ (`OdeSys::fmass`) the implicit stage solves
/// $M (y_i - y_{expl}) - \Delta t \thinspace a_{ii} f(t_i, y_i) = 0$ with Newton matrix
/// $M - \Delta t \thinspace a_{ii} J$, and the stage derivative is
/// `k[i] = (y_i - y_expl) / (dt*a_ii)`. Explicit stages use `k[i] = M^{-1} f`
/// computed with GMRES. See the module documentation.
///
/// Lifetime `'jac` is the lifetime of the ODE system (governs `ShiftedLinOp`).
/// `y0` may have any lifetime shorter than `'jac`; it is cloned on entry.
fn dirk_step<'jac>(
    sys: &'jac dyn OdeSys<'jac>,
    t: f64,
    y0: MatRef<'_, f64>,
    dt: f64,
    bt: &ImplicitBT,
    tol_nlin: f64,
    tol_lin: f64,
    iters_nlin: usize,
    iters_lin: usize,
) -> Result<StepResult<f64, Mat<f64>>, StepError> {
    let s = bt.s;
    // Stage derivatives k[i] = f(t + c[i]*dt, y_i)
    let mut k: Vec<Mat<f64>> = Vec::with_capacity(s);

    for i in 0..s {
        // explicit accumulation: y_expl = y0 + dt * \sum_{j<i} a[i][j]*k[j]
        let mut y_expl: Mat<f64> = y0.to_owned();
        for j in 0..i {
            y_expl = y_expl.as_ref() + faer::Scale(dt * bt.a[i][j]) * k[j].as_ref();
        }

        let a_ii = bt.a[i][i];
        let t_i = t + bt.c[i] * dt;

        // Optional mass matrix M(t_i). `None` means M = I.
        let mass = sys.fmass(t_i);

        let k_i: Mat<f64> = if a_ii == 0.0 {
            // Explicit stage: k_i = M^{-1} f(t_i, y_expl)
            let f_i = sys.frhs(t_i, y_expl.as_ref());
            match &mass {
                None => f_i,
                Some(m) => solve_mass(m.as_ref(), f_i.as_ref(), tol_lin, iters_lin)?,
            }
        } else {
            print!("Implicit stage: {} ", i + 1);
            // Implicit stage
            // Solve  g(y_i) = M (y_i - y_expl) - dt*a_ii*f(t_i, y_i) = 0
            // dg/dy_i = M - dt*a_ii * J_f(t_i, y_i)
            //         == fjac_shifted(t_i, y_i, scale=-dt*a_ii, gamma=1.0)
            let scale_ii = -dt * a_ii;

            // HRTB on the input MatRef so `jac_newton` can call the closure
            // with its internal iteration variable.
            let gfn: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> Mat<f64> = &|t_arg, y_i| {
                let dy = y_i.as_ref() - y_expl.as_ref();
                let m_dy = match &mass {
                    None => dy,
                    Some(m) => apply_mass(m.as_ref(), dy.as_ref()),
                };
                m_dy - faer::Scale(dt * a_ii) * sys.frhs(t_arg, y_i)
            };

            // Return lifetime is 'jac (tied to `sys`), independent of input 'c.
            let gfn_jac: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> ShiftedLinOp<'jac> =
                &|t_arg, y_i| sys.fjac_shifted(t_arg, y_i, scale_ii, Some(1.0));

            // Newton initial guess = y_expl (explicit accumulation for this stage).
            let y_i = jac_newton(
                t_i,
                y_expl.as_ref(),
                gfn,
                gfn_jac,
                tol_nlin,
                tol_lin,
                iters_nlin,
                iters_lin,
            )?;

            // Stage derivative k_i = M^{-1} f(t_i, y_i). With a mass matrix this is
            // obtained from the stage equation without solving with M:
            // k_i = (y_i - y_expl) / (dt*a_ii)
            match &mass {
                None => sys.frhs(t_i, y_i.as_ref()),
                Some(_) => faer::Scale(1.0 / (dt * a_ii)) * (y_i.as_ref() - y_expl.as_ref()),
            }
        };

        k.push(k_i);
    }

    // final accumulation: y_{n+1} = y0 + dt * \sum_i ( b[i]*k[i] )
    let mut y_new: Mat<f64> = y0.to_owned();
    for i in 0..s {
        y_new = y_new.as_ref() + faer::Scale(dt * bt.b[i]) * k[i].as_ref();
    }
    Ok(StepResult::new(t + dt, dt, y_new, None))
}

/// Generic single-step DIRK / SDIRK integrator defined by an [`ImplicitBT`] tableau.
///
/// Works with any fully-implicit or ESDIRK tableau: Backward Euler,
/// Crank-Nicolson, SDIRK22, SDIRK32, SDIRK32 (Norsett), SDIRK33, etc.
///
/// Each implicit stage is solved by Newton-Krylov with at most 50 nonlinear
/// iterations and 1000 linear (GMRES) iterations.
///
/// Systems with a mass matrix (`OdeSys::fmass`), $M y^\prime = f$, are supported.
/// Tableaux with an explicit stage (such as Crank-Nicolson) require a nonsingular
/// $M$. See the module documentation.
pub struct DirkIntegrator<'a> {
    bt: ImplicitBT,
    t: f64,
    y: Mat<f64>,
    tol_lin: f64,
    tol_nlin: f64,
    iters_lin: usize,
    iters_nlin: usize,
    phantom: PhantomData<&'a ()>,
}

impl<'a> DirkIntegrator<'a> {
    /// Set the initial conditions and create a DIRK integrator.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state
    /// * `bt` - Butcher tableau of the method, for example
    ///   [`ImplicitBT::sdirk32`]
    /// * `tol_lin` - tolerance of the GMRES linear solver, must be positive
    /// * `tol_nlin` - tolerance of the Newton solver on the 2-norm of the stage
    ///   residual, must be positive
    ///
    /// # Panics
    ///
    /// Panics if `tol_lin` or `tol_nlin` is not positive.
    pub fn new(t0: f64, y0: MatRef<'_, f64>, bt: ImplicitBT, tol_lin: f64, tol_nlin: f64) -> Self {
        assert!(tol_nlin > 0.);
        assert!(tol_lin > 0.);
        Self {
            bt,
            t: t0,
            y: y0.to_owned(),
            tol_lin: tol_lin,
            tol_nlin: tol_nlin,
            iters_lin: 1000,
            iters_nlin: 50,
            phantom: Default::default(),
        }
    }
}

impl<'a> IntegrateSys<'a> for DirkIntegrator<'a> {
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        println!("\nDIRK step, t: {:?}, dt: {:?}", self.t, dt);
        let clock = std::time::Instant::now();
        let res = dirk_step(
            sys,
            self.t,
            self.y.as_ref(),
            dt,
            &self.bt,
            self.tol_nlin,
            self.tol_lin,
            self.iters_nlin,
            self.iters_lin,
        );
        println!("DIRK step time (s): {}", clock.elapsed().as_secs_f64());
        res
    }

    fn time(&self) -> Self::TimeType {
        self.t
    }

    fn state(&self) -> Self::SysStateType {
        self.y.clone()
    }

    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>) {
        self.t = s.t;
        self.y = s.y;
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.t = t0;
        self.y = y0;
    }
}

/// BDF linear multistep integrator.
///
/// `order = 1` - BDF1 (Backward Euler); delegates to `dirk_step` with
///               `ImplicitBT::implicit_euler()`.
///
/// `order = 2` - BDF2; requires solution history. Bootstraps with BDF1 on the
///               first step when history is not yet full.  Cannot be expressed
///               as a Butcher tableau (multistep method).
///
/// BDF2 assumes a constant step size and uses
///
/// $$ y_{n+1} = \frac{4}{3} y_n - \frac{1}{3} y_{n-1} + \frac{2}{3} \Delta t \thinspace f(t_{n+1}, y_{n+1}) $$
///
/// Each step is solved by Newton-Krylov with at most 50 nonlinear iterations
/// and 1000 linear (GMRES) iterations.
///
/// Systems with a mass matrix (`OdeSys::fmass`), $M y^\prime = f$, are supported,
/// including a singular $M$. The BDF2 Newton matrix is
/// $M - \frac{2}{3} \Delta t \thinspace J$. See the module documentation.
pub struct BdfIntegrator<'a> {
    order: usize,
    t: f64,
    /// History: index 0 = y_n (most recent), index 1 = y_{n-1}
    y_hist: VecDeque<Mat<f64>>,
    tol_lin: f64,
    tol_nlin: f64,
    iters_lin: usize,
    iters_nlin: usize,
    phantom: PhantomData<&'a ()>,
}

impl<'a> BdfIntegrator<'a> {
    /// Set the initial conditions and create a BDF integrator.
    ///
    /// # Arguments
    ///
    /// * `t0` - initial time
    /// * `y0` - initial state
    /// * `order` - order of the method, 1 (BDF1) or 2 (BDF2)
    /// * `tol_lin` - tolerance of the GMRES linear solver, must be positive
    /// * `tol_nlin` - tolerance of the Newton solver on the 2-norm of the
    ///   residual, must be positive
    ///
    /// # Panics
    ///
    /// Panics if `tol_lin` or `tol_nlin` is not positive. An unsupported
    /// `order` is not detected here, but causes a panic when stepping.
    pub fn new(t0: f64, y0: MatRef<'_, f64>, order: usize, tol_lin: f64, tol_nlin: f64) -> Self {
        assert!(tol_nlin > 0.);
        assert!(tol_lin > 0.);
        let mut y_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        Self {
            order,
            t: t0,
            y_hist,
            tol_lin: tol_lin,
            tol_nlin: tol_nlin,
            iters_lin: 1000,
            iters_nlin: 50,
            phantom: Default::default(),
        }
    }

    /// BDF1 delegates to ImplicitBT::implicit_euler()
    fn step_order_1<'b>(
        &self,
        sys: &'b dyn OdeSys<'b>,
        dt: f64,
    ) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        dirk_step(
            sys,
            self.t,
            self.y_hist[0].as_ref(),
            dt,
            &ImplicitBT::implicit_euler(),
            self.tol_nlin,
            self.tol_lin,
            self.iters_nlin,
            self.iters_lin,
        )
    }

    /// BDF2: linear multistep
    fn step_order_2<'b>(
        &self,
        sys: &'b dyn OdeSys<'b>,
        dt: f64,
    ) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        let t = self.t;
        let y0 = self.y_hist[0].as_ref(); // y_n
        let y1 = self.y_hist[1].as_ref(); // y_{n-1}

        // BDF2 formula:
        //   M (y_{n+1} - (4/3)*y_n + (1/3)*y_{n-1}) = (2/3)*dt*f(t+dt, y_{n+1})
        // (M = I when the system has no mass matrix).
        //
        // Nonlinear residual:
        //   g(y) = M (y - (4/3)*y_n + (1/3)*y_{n-1}) - (2/3)*dt*f(t+dt, y) = 0
        //
        // Jacobian of g:
        //   dg/dy = gamma*M - (2/3)*dt*J_f  ==  fjac_shifted(scale=-2dt/3, gamma=1)
        let scale = -(2.0 / 3.0) * dt;
        let mass = sys.fmass(t + dt);

        let gfn: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> Mat<f64> = &|t_arg, y| {
            let dy = y.as_ref() - faer::Scale(4.0 / 3.0) * y0 + faer::Scale(1.0 / 3.0) * y1;
            let m_dy = match &mass {
                None => dy,
                Some(m) => apply_mass(m.as_ref(), dy.as_ref()),
            };
            m_dy - faer::Scale((2.0 / 3.0) * dt) * sys.frhs(t_arg, y)
        };

        let gfn_jac: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> ShiftedLinOp<'b> =
            &|t_arg, y| sys.fjac_shifted(t_arg, y, scale, Some(1.0));

        let y_new = jac_newton(
            t + dt,
            y0,
            gfn,
            gfn_jac,
            self.tol_nlin,
            self.tol_lin,
            self.iters_nlin,
            self.iters_lin,
        )?;
        Ok(StepResult::new(t + dt, dt, y_new, None))
    }
}

impl<'a> IntegrateSys<'a> for BdfIntegrator<'a> {
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        println!("\nBDF step, t: {:?}, dt: {:?}", self.t, dt);
        let clock = std::time::Instant::now();
        let res = match self.order {
            1 => self.step_order_1(sys, dt),
            2 => {
                if self.y_hist.len() >= 2 {
                    self.step_order_2(sys, dt)
                } else {
                    // Bootstrap: not enough history yet, use BDF1
                    self.step_order_1(sys, dt)
                }
            }
            _ => panic!("BdfIntegrator: unsupported order {}", self.order),
        };
        println!("BDF step time (s): {}", clock.elapsed().as_secs_f64());
        res
    }

    fn time(&self) -> Self::TimeType {
        self.t
    }

    fn state(&self) -> Self::SysStateType {
        self.y_hist[0].to_owned()
    }

    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>) {
        self.t = s.t;
        self.y_hist.push_front(s.y);
        if self.y_hist.len() > self.order {
            self.y_hist.pop_back();
        }
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.y_hist.push_front(y0);
        self.t = t0;
    }
}

#[cfg(test)]
mod test_implicit {
    use super::*;
    use crate::ode_rk::RkIntegrator;
    use crate::test_common::*;

    /// Test parameters: Lotka-Volterra y0=[5,4], 10 steps * dt=0.01, tf=0.1
    const DT: f64 = 0.01;
    const N_STEPS: usize = 10;
    const T_END: f64 = DT * N_STEPS as f64; // 0.1 s

    /// Convenience stepper used with `dyn IntegrateSys` (for integrators that
    /// carry a phantom lifetime, e.g. `BdfIntegrator` and `DirkIntegrator`).
    fn run_steps<'a>(
        solver: &mut dyn IntegrateSys<'a, TimeType = f64, SysStateType = Mat<f64>>,
        sys: &'a dyn OdeSys<'a>,
        dt: f64,
        n: usize,
    ) {
        for _ in 0..n {
            let res = solver.step(sys, dt).unwrap();
            solver.accept_step(res);
        }
    }

    /// Generate the RK4 reference solution on Lotka-Volterra at t = T_END.
    ///
    /// RK4 is 4th order; global error at t=0.1 with dt=0.01 is
    /// O(dt^4) ~= 1e-8, negligible compared with the 1st-3rd order implicit
    /// errors being tested.
    fn rk4_reference() -> Mat<f64> {
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0_f64,], [4.0_f64,]];
        let mut rk4 = RkIntegrator::new(0.0, y0.as_ref(), 4);
        for _ in 0..N_STEPS {
            let res = rk4.step(&sys, DT).unwrap();
            rk4.accept_step(res);
        }
        println!("RK4 reference at t={T_END}: y = {:?}", rk4.state());
        rk4.state()
    }

    /// Assert `y` is within `tol_rel` (relative) of the RK4 reference `y_ref`.
    ///
    /// Scale is max(|y_ref[i]|, 1e-8) to handle near-zero components.
    fn assert_close_to_rk4(label: &str, y: &Mat<f64>, y_ref: &Mat<f64>, tol_rel: f64) {
        for row in 0..y_ref.nrows() {
            let diff = (y[(row, 0)] - y_ref[(row, 0)]).abs();
            let scale = y_ref[(row, 0)].abs().max(1e-8);
            let tol = tol_rel * scale;
            assert!(
                diff < tol,
                "{label} component[{row}]: got {:.8}, RK4={:.8}, \
                 rel-err={:.2e} exceeds tol {tol_rel:.2e}",
                y[(row, 0)],
                y_ref[(row, 0)],
                diff / scale
            );
        }
    }

    /// BDF1 with finite-difference Jacobian (JFNK path) vs RK4 baseline.
    /// BDF1 is 1st order; 5 % relative tolerance at dt=0.01.
    #[test]
    fn test_bdf1_jfnk() {
        let y_rk4 = rk4_reference();
        let sys = TestLvFdSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = BdfIntegrator::new(0.0, y0.as_ref(), 1, 1e-12, 1e-12);
        for _ in 0..N_STEPS {
            let res = solver.step(&sys, DT).unwrap();
            solver.accept_step(res);
        }
        let y = solver.state();
        println!("BDF1 JFNK at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("BDF1 JFNK", &y, &y_rk4, 5e-2);
    }

    /// BDF2 with exact analytic Jacobian vs RK4 baseline.
    /// BDF2 is 2nd order; 0.5 % relative tolerance at dt=0.01.
    #[test]
    fn test_bdf2_nk() {
        let y_rk4 = rk4_reference();
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = BdfIntegrator::new(0.0, y0.as_ref(), 2, 1e-12, 1e-12);
        for _ in 0..N_STEPS {
            let res = solver.step(&sys, DT).unwrap();
            solver.accept_step(res);
        }
        let y = solver.state();
        println!("BDF2 NK at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("BDF2 NK", &y, &y_rk4, 5e-3);
    }

    /// SDIRK22 with finite-difference Jacobian (JFNK) vs RK4.
    /// 2nd order; 0.5 % relative tolerance at dt=0.01.
    #[test]
    fn test_sdirk22_fd() {
        let y_rk4 = rk4_reference();
        let sys = TestLvFdSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk22(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK22 FD at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK22 FD", &y, &y_rk4, 5e-3);
    }

    /// SDIRK22 with exact analytic Jacobian vs RK4.
    #[test]
    fn test_sdirk22_exact_jac() {
        let y_rk4 = rk4_reference();
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk22(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK22 exact Jac at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK22 exact Jac", &y, &y_rk4, 5e-3);
    }

    /// SDIRK32 (L-stable, gamma=1/4) with finite-difference Jacobian vs RK4.
    /// 2nd order; 0.5 % relative tolerance at dt=0.01.
    #[test]
    fn test_sdirk32_fd() {
        let y_rk4 = rk4_reference();
        let sys = TestLvFdSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK32 FD at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK32 FD", &y, &y_rk4, 5e-3);
    }

    /// SDIRK32 (L-stable default) with exact analytic Jacobian vs RK4.
    #[test]
    fn test_sdirk32_exact_jac() {
        let y_rk4 = rk4_reference();
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK32 exact Jac at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK32 exact Jac", &y, &y_rk4, 5e-3);
    }

    /// SDIRK32 Norsett variant (gamma=(3-sqrt(3))/6) with exact analytic Jacobian vs RK4.
    #[test]
    fn test_sdirk32_norsett_exact_jac() {
        let y_rk4 = rk4_reference();
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(
            0.0,
            y0.as_ref(),
            ImplicitBT::sdirk32_norsett(),
            1e-12,
            1e-12,
        );
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK32 Norsett at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK32 Norsett", &y, &y_rk4, 5e-3);
    }

    /// SDIRK33 Alexander (1977) - 3rd order - vs RK4.
    /// 3rd order; 0.05 % relative tolerance at dt=0.01 (tighter than 2nd order).
    #[test]
    fn test_sdirk33_exact_jac() {
        let y_rk4 = rk4_reference();
        let sys = TestLvSys::new();
        let y0 = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk33(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK33 Alexander at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK33 Alexander", &y, &y_rk4, 5e-4);
    }

    /// All implicit DIRK variants are tested against the RK4 reference solution
    /// at t=0.1 (10 steps x dt=0.01) on the Lotka-Volterra system.
    ///
    /// Tolerance reflects the expected global error for each method's order:
    ///   order 1  (Backward Euler):  O(dt)    -> 5 %
    ///   order 2  (CN, SDIRK22/32):  O(dt^2)  -> 0.5 %
    ///   order 3  (SDIRK33):         O(dt^3)  -> 0.05 %
    ///
    /// RK4 global error is O(dt^4) ~= 1e-8 at t=0.1, negligible as reference.
    #[test]
    fn test_implicit_methods_vs_rk4() {
        let y_rk4 = rk4_reference();
        println!("RK4 reference at t={T_END}: {:?}", y_rk4);

        // (method name, ImplicitBT, expected order, relative tolerance)
        let methods: &[(&str, ImplicitBT, usize, f64)] = &[
            ("ImplicitEuler", ImplicitBT::implicit_euler(), 1, 5e-2),
            ("CrankNicolson", ImplicitBT::crank_nicolson(), 2, 5e-3),
            ("SDIRK22", ImplicitBT::sdirk22(), 2, 5e-3),
            ("SDIRK32", ImplicitBT::sdirk32(), 2, 5e-3),
            ("SDIRK32_Norsett", ImplicitBT::sdirk32_norsett(), 2, 5e-3),
            ("SDIRK33", ImplicitBT::sdirk33(), 3, 5e-4),
        ];

        for (name, bt, order, tol_rel) in methods {
            let sys = TestLvSys::new();
            let y0 = faer::mat![[5.0_f64,], [4.0_f64,]];
            let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), bt.clone(), 1e-12, 1e-12);
            run_steps(&mut solver, &sys, DT, N_STEPS);
            let y = solver.state();

            // Compute relative error vs RK4 for reporting
            let rel_err: Vec<f64> = (0..y_rk4.nrows())
                .map(|r| {
                    let diff = (y[(r, 0)] - y_rk4[(r, 0)]).abs();
                    let scale = y_rk4[(r, 0)].abs().max(1e-8);
                    diff / scale
                })
                .collect();
            println!(
                "{name} (order {order}): y={:?}  rel-err={:?}  tol={tol_rel:.2e}",
                (0..y.nrows())
                    .map(|r| format!("{:.6}", y[(r, 0)]))
                    .collect::<Vec<_>>(),
                rel_err
                    .iter()
                    .map(|e| format!("{e:.2e}"))
                    .collect::<Vec<_>>()
            );

            assert_close_to_rk4(name, &y, &y_rk4, *tol_rel);
        }
    }

    // ---------------------------------------------------------------------
    // Mass matrix tests
    // ---------------------------------------------------------------------

    /// Lotka-Volterra with constant diagonal mass matrix: `M y' = f(y)`.
    struct MassLvSys {
        m: Mat<f64>,
    }

    impl<'a> OdeSys<'a> for MassLvSys {
        fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
            crate::ode_utils::lv_sys_rhs(t, x)
        }
        fn fjac<'b>(&'a self, t: f64, x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
            Box::new(get_fd_jac(self, t, x))
        }
        fn fmass(&'a self, _t: f64) -> Option<Box<dyn LinOp<f64> + 'a>> {
            Some(Box::new(self.m.clone()))
        }
    }

    /// The same problem with `M^{-1}` folded into the rhs: `y' = M^{-1} f(y)`, no mass matrix.
    struct ScaledLvSys {
        m_inv_diag: Vec<f64>,
    }

    impl<'a> OdeSys<'a> for ScaledLvSys {
        fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
            let mut f = crate::ode_utils::lv_sys_rhs(t, x);
            for i in 0..f.nrows() {
                f[(i, 0)] *= self.m_inv_diag[i];
            }
            f
        }
        fn fjac<'b>(&'a self, t: f64, x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
            Box::new(get_fd_jac(self, t, x))
        }
    }

    /// Every implicit method must solve `M y' = f` to the same result as the
    /// equivalent system `y' = M^{-1} f` (including the explicit first stage of
    /// Crank-Nicolson, which needs a solve with `M`).
    #[test]
    fn test_mass_matrix_matches_scaled_system() {
        let mass_sys = MassLvSys {
            m: faer::mat![[2.0, 0.0], [0.0, 3.0]],
        };
        let scaled_sys = ScaledLvSys {
            m_inv_diag: vec![0.5, 1.0 / 3.0],
        };
        let y0 = faer::mat![[5.0_f64,], [4.0_f64,]];

        let mut integrators: Vec<(&str, Box<dyn Fn() -> Box<dyn IntegrateSys<'static, TimeType = f64, SysStateType = Mat<f64>>>>)> = Vec::new();
        for (name, order) in [("bdf1", 1), ("bdf2", 2)] {
            let y0 = y0.clone();
            integrators.push((
                name,
                Box::new(move || Box::new(BdfIntegrator::new(0.0, y0.as_ref(), order, 1e-12, 1e-12))),
            ));
        }
        for (name, bt) in [
            ("cn", ImplicitBT::crank_nicolson()),
            ("sdirk22", ImplicitBT::sdirk22()),
            ("sdirk32", ImplicitBT::sdirk32()),
            ("sdirk32_norsett", ImplicitBT::sdirk32_norsett()),
            ("sdirk33", ImplicitBT::sdirk33()),
        ] {
            let y0 = y0.clone();
            integrators.push((
                name,
                Box::new(move || {
                    Box::new(DirkIntegrator::new(0.0, y0.as_ref(), bt.clone(), 1e-12, 1e-12))
                }),
            ));
        }

        for (name, make) in integrators {
            let mut with_mass = make();
            let mut scaled = make();
            for _ in 0..N_STEPS {
                let r = with_mass.step(&mass_sys, DT).unwrap();
                with_mass.accept_step(r);
                let r = scaled.step(&scaled_sys, DT).unwrap();
                scaled.accept_step(r);
            }
            let diff = (with_mass.state() - scaled.state()).norm_max();
            println!("{name}: |y_mass - y_scaled|_max = {diff:.3e}");
            assert!(diff < 1e-7, "{name}: mass matrix result differs by {diff:.3e}");
        }
    }

    /// Index-1 DAE `y1' = -y1`, `0 = y1 - y2` written as `M y' = A y` with the
    /// singular mass matrix `M = diag(1, 0)`. The algebraic constraint must hold
    /// after every step, and `y1` must decay as `exp(-t)`.
    struct DaeSys {
        a: Mat<f64>,
        m: Mat<f64>,
    }

    impl<'a> OdeSys<'a> for DaeSys {
        fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
            self.a.as_ref() * x
        }
        fn fjac<'b>(&'a self, _t: f64, _x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
            Box::new(self.a.clone())
        }
        fn fmass(&'a self, _t: f64) -> Option<Box<dyn LinOp<f64> + 'a>> {
            Some(Box::new(self.m.clone()))
        }
    }

    #[test]
    fn test_singular_mass_matrix_dae() {
        let sys = DaeSys {
            a: faer::mat![[-1.0, 0.0], [1.0, -1.0]],
            m: faer::mat![[1.0, 0.0], [0.0, 0.0]],
        };
        let y0 = faer::mat![[1.0_f64,], [1.0_f64,]];

        // (name, integrator). All stages are implicit so a singular M is allowed.
        let mut solvers: Vec<(&str, Box<dyn IntegrateSys<'static, TimeType = f64, SysStateType = Mat<f64>>>)> = vec![
            ("bdf1", Box::new(BdfIntegrator::new(0.0, y0.as_ref(), 1, 1e-12, 1e-12))),
            ("bdf2", Box::new(BdfIntegrator::new(0.0, y0.as_ref(), 2, 1e-12, 1e-12))),
            ("sdirk22", Box::new(DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk22(), 1e-12, 1e-12))),
            ("sdirk32", Box::new(DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32(), 1e-12, 1e-12))),
            ("sdirk33", Box::new(DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk33(), 1e-12, 1e-12))),
        ];
        for (name, solver) in solvers.iter_mut() {
            for _ in 0..N_STEPS {
                let r = solver.step(&sys, DT).unwrap();
                solver.accept_step(r);
            }
            let y = solver.state();
            let constraint = (y[(0, 0)] - y[(1, 0)]).abs();
            let err = (y[(0, 0)] - (-T_END).exp()).abs();
            println!("{name}: constraint = {constraint:.3e}, y1 err = {err:.3e}");
            assert!(constraint < 1e-8, "{name}: algebraic constraint violated: {constraint:.3e}");
            assert!(err < 1e-3, "{name}: y1 error {err:.3e}");
        }
    }
}
