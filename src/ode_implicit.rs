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

/// Implicit time integration:
///   - Generic DIRK / SDIRK via Butcher tableau  (`DirkIntegrator`)
///   - BDF1 and BDF2 linear multistep methods     (`BdfIntegrator`)
///
use faer::prelude::*;
use crate::ode_sys::*;
use crate::newton::*;
use crate::tableau_implicit::ImplicitBT;
use std::marker::PhantomData;
use std::collections::VecDeque;


/// Advance `y' = f(t,y)` by one step `dt` using the implicit Butcher tableau `bt`.
///
/// For each stage `i`:
///
/// 1. Build the explicit accumulation
///    `y_expl = y0 + dt * \sum_{j < i} a[i][j] · k[j]`
///
/// 2. Explicit stage. (`a[i][i] == 0`):
///    `k[i] = f(t + c[i]*dt, y_expl)`
///
/// 3. Implicit stage. (`a[i][i] != 0`):
///    Solve  `g(y_i) = y_i − y_expl − dt*a[i][i]*f(t_i, y_i) = 0`
///    using Newton–Krylov, starting from `y_expl`, with Jacobian
///    `I − dt*a[i][i]*J_f ≡ fjac_shifted(t_i, y_i, −dt*a[i][i], γ=1)`.
///    Then `k[i] = f(t_i, y_i)`.
///
/// Finally `y_{n+1} = y0 + dt * \sum_i b[i] · k[i]`.
///
/// Lifetime `'jac` is the lifetime of the ODE system (governs `ShiftedLinOp`).
/// `y0` may have any lifetime shorter than `'jac`; it is cloned on entry.
fn dirk_step<'jac>(
    sys:       &'jac dyn OdeSys<'jac>,
    t:         f64,
    y0:        MatRef<'_, f64>,
    dt:        f64,
    bt:        &ImplicitBT,
    tol_nlin:  f64,
    tol_lin:   f64,
    iters_nlin: usize,
    iters_lin:  usize,
) -> Result<StepResult<f64, Mat<f64>>, StepError> {
    let s = bt.s;
    // Stage derivatives k[i] = f(t + c[i]*dt, y_i)
    let mut k: Vec<Mat<f64>> = Vec::with_capacity(s);

    for i in 0..s {
        // explicit accumulation: y_expl = y0 + dt * \sum_{j<i} a[i][j]*k[j]
        let mut y_expl: Mat<f64> = y0.to_owned();
        for j in 0..i {
            y_expl = y_expl.as_ref()
                + faer::Scale(dt * bt.a[i][j]) * k[j].as_ref();
        }

        let a_ii = bt.a[i][i];
        let t_i  = t + bt.c[i] * dt;

        let k_i: Mat<f64> = if a_ii == 0.0 {
            // Explicit stage
            sys.frhs(t_i, y_expl.as_ref())

        } else {
            // Implicit stage
            // Solve  g(y_i) = y_i - y_expl - dt*a_ii*f(t_i, y_i) = 0
            // dg/dy_i = I - dt*a_ii * J_f(t_i, y_i)
            //         ≡ fjac_shifted(t_i, y_i, scale=-dt*a_ii, gamma=1.0)
            let scale_ii = -dt * a_ii;

            // HRTB on the input MatRef so `jac_newton` can call the closure
            // with its internal iteration variable.
            let gfn: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> Mat<f64> =
                &|t_arg, y_i| {
                    y_i.as_ref()
                        - y_expl.as_ref()
                        - faer::Scale(dt * a_ii) * sys.frhs(t_arg, y_i)
                };

            // Return lifetime is 'jac (tied to `sys`), independent of input 'c.
            let gfn_jac: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> ShiftedLinOp<'jac> =
                &|t_arg, y_i| sys.fjac_shifted(t_arg, y_i, scale_ii, Some(1.0));

            // Newton initial guess = y_expl (explicit accumulation for this stage).
            let y_i = jac_newton(
                t_i, y_expl.as_ref(),
                gfn, gfn_jac,
                tol_nlin, tol_lin, iters_nlin, iters_lin,
            )?;

            // Stage derivative
            sys.frhs(t_i, y_i.as_ref())
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
/// Crank–Nicolson, SDIRK22, SDIRK32, SDIRK32 (Norsett), SDIRK33, ect.
///
pub struct DirkIntegrator<'a> {
    bt: ImplicitBT,
    t:  f64,
    y:  Mat<f64>,
    tol_lin:    f64,
    tol_nlin:   f64,
    iters_lin:  usize,
    iters_nlin: usize,
    phantom: PhantomData<&'a ()>,
}

impl<'a> DirkIntegrator<'a> {
    pub fn new(t0: f64, y0: MatRef<'_, f64>, bt: ImplicitBT, tol_lin: f64, tol_nlin: f64) -> Self {
        assert!(tol_nlin > 0.);
        assert!(tol_lin > 0.);
        Self {
            bt,
            t: t0,
            y: y0.to_owned(),
            tol_lin:    tol_lin,
            tol_nlin:   tol_nlin,
            iters_lin:  1000,
            iters_nlin: 50,
            phantom: Default::default(),
        }
    }
}

impl<'a> IntegrateSys<'a> for DirkIntegrator<'a> {
    type TimeType     = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        dirk_step(
            sys, self.t, self.y.as_ref(), dt, &self.bt,
            self.tol_nlin, self.tol_lin, self.iters_nlin, self.iters_lin,
        )
    }

    fn time(&self) -> Self::TimeType { self.t }

    fn state(&self) -> Self::SysStateType { self.y.clone() }

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
/// `order = 1` — BDF1 (Backward Euler); delegates to `dirk_step` with
///               `ImplicitBT::implicit_euler()`.
///
/// `order = 2` — BDF2; requires solution history. Bootstraps with BDF1 on the
///               first step when history is not yet full.  Cannot be expressed
///               as a Butcher tableau (multistep method).
///
pub struct BdfIntegrator<'a> {
    order: usize,
    t:     f64,
    /// History: index 0 = y_n (most recent), index 1 = y_{n-1}
    y_hist: VecDeque<Mat<f64>>,
    tol_lin:    f64,
    tol_nlin:   f64,
    iters_lin:  usize,
    iters_nlin: usize,
    phantom: PhantomData<&'a ()>,
}

impl<'a> BdfIntegrator<'a> {
    pub fn new(t0: f64, y0: MatRef<'_, f64>, order: usize, tol_lin: f64, tol_nlin: f64) -> Self {
        assert!(tol_nlin > 0.);
        assert!(tol_lin > 0.);
        let mut y_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        Self {
            order,
            t: t0,
            y_hist,
            tol_lin:    tol_lin,
            tol_nlin:   tol_nlin,
            iters_lin:  1000,
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
            sys, self.t, self.y_hist[0].as_ref(), dt,
            &ImplicitBT::implicit_euler(),
            self.tol_nlin, self.tol_lin, self.iters_nlin, self.iters_lin,
        )
    }

    /// BDF2: linear multistep
    fn step_order_2<'b>(
        &self,
        sys: &'b dyn OdeSys<'b>,
        dt: f64,
    ) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        let t  = self.t;
        let y0 = self.y_hist[0].as_ref();   // y_n
        let y1 = self.y_hist[1].as_ref();   // y_{n-1}

        // BDF2 formula:
        //   y_{n+1} = (4/3)*y_n − (1/3)*y_{n-1} + (2/3)*dt*f(t+dt, y_{n+1})
        //
        // Nonlinear residual:
        //   g(y) = y − (4/3)*y_n + (1/3)*y_{n-1} − (2/3)*dt*f(t+dt, y) = 0
        //
        // Jacobian of g:
        //   dg/dy = gamma*I − (2/3)*dt*J_f  ≡  fjac_shifted(scale=−2dt/3, gamma=1)
        let scale = -(2.0 / 3.0) * dt;

        let gfn: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> Mat<f64> =
            &|t_arg, y| {
                y.as_ref()
                    - faer::Scale(4.0 / 3.0) * y0
                    + faer::Scale(1.0 / 3.0) * y1
                    - faer::Scale((2.0 / 3.0) * dt) * sys.frhs(t_arg, y)
            };

        let gfn_jac: &dyn for<'c> Fn(f64, MatRef<'c, f64>) -> ShiftedLinOp<'b> =
            &|t_arg, y| sys.fjac_shifted(t_arg, y, scale, Some(1.0));

        let y_new = jac_newton(
            t + dt, y0,
            gfn, gfn_jac,
            self.tol_nlin, self.tol_lin, self.iters_nlin, self.iters_lin,
        )?;
        Ok(StepResult::new(t + dt, dt, y_new, None))
    }

}

impl<'a> IntegrateSys<'a> for BdfIntegrator<'a> {
    type TimeType     = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(
        &mut self,
        sys: &'b dyn OdeSys<'b>,
        dt: Self::TimeType,
    ) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        match self.order {
            1 => self.step_order_1(sys, dt),
            2 => {
                if self.y_hist.len() >= 2 {
                    self.step_order_2(sys, dt)
                } else {
                    // Bootstrap: not enough history yet, use BDF1
                    self.step_order_1(sys, dt)
                }
            },
            _ => panic!("BdfIntegrator: unsupported order {}", self.order),
        }
    }

    fn time(&self) -> Self::TimeType { self.t }

    fn state(&self) -> Self::SysStateType { self.y_hist[0].to_owned() }

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
    use crate::test_common::*;
    use crate::ode_rk::RkIntegrator;
    use super::*;

    /// Test parameters: Lotka–Volterra y0=[5,4], 10 steps * dt=0.01, tf=0.1
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

    /// Generate the RK4 reference solution on Lotka–Volterra at t = T_END.
    ///
    /// RK4 is 4th order; global error at t=0.1 with dt=0.01 is
    /// O(dt^4) ≈ 1e-8, negligible compared with the 1st–3rd order implicit
    /// errors being tested.
    fn rk4_reference() -> Mat<f64> {
        let sys = TestLvSys::new();
        let y0  = faer::mat![[5.0_f64,], [4.0_f64,]];
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
            let diff  = (y[(row, 0)] - y_ref[(row, 0)]).abs();
            let scale = y_ref[(row, 0)].abs().max(1e-8);
            let tol   = tol_rel * scale;
            assert!(
                diff < tol,
                "{label} component[{row}]: got {:.8}, RK4={:.8}, \
                 rel-err={:.2e} exceeds tol {tol_rel:.2e}",
                y[(row, 0)], y_ref[(row, 0)],
                diff / scale
            );
        }
    }

    /// BDF1 with finite-difference Jacobian (JFNK path) vs RK4 baseline.
    /// BDF1 is 1st order; 5 % relative tolerance at dt=0.01.
    #[test]
    fn test_bdf1_jfnk() {
        let y_rk4 = rk4_reference();
        let sys   = TestLvFdSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
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
        let sys   = TestLvSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
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
        let sys   = TestLvFdSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
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
        let sys   = TestLvSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk22(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK22 exact Jac at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK22 exact Jac", &y, &y_rk4, 5e-3);
    }

    /// SDIRK32 (L-stable, γ=1/4) with finite-difference Jacobian vs RK4.
    /// 2nd order; 0.5 % relative tolerance at dt=0.01.
    #[test]
    fn test_sdirk32_fd() {
        let y_rk4 = rk4_reference();
        let sys   = TestLvFdSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
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
        let sys   = TestLvSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK32 exact Jac at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK32 exact Jac", &y, &y_rk4, 5e-3);
    }

    /// SDIRK32 Norsett variant (γ=(3−sqrt(3))/6) with exact analytic Jacobian vs RK4.
    #[test]
    fn test_sdirk32_norsett_exact_jac() {
        let y_rk4 = rk4_reference();
        let sys   = TestLvSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32_norsett(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK32 Norsett at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK32 Norsett", &y, &y_rk4, 5e-3);
    }

    /// SDIRK33 Alexander (1977) — 3rd order — vs RK4.
    /// 3rd order; 0.05 % relative tolerance at dt=0.01 (tighter than 2nd order).
    #[test]
    fn test_sdirk33_exact_jac() {
        let y_rk4 = rk4_reference();
        let sys   = TestLvSys::new();
        let y0    = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk33(), 1e-12, 1e-12);
        run_steps(&mut solver, &sys, DT, N_STEPS);
        let y = solver.state();
        println!("SDIRK33 Alexander at t={T_END}: y = {:?}", y);
        assert_close_to_rk4("SDIRK33 Alexander", &y, &y_rk4, 5e-4);
    }

    /// All implicit DIRK variants are tested against the RK4 reference solution
    /// at t=0.1 (10 steps × dt=0.01) on the Lotka–Volterra system.
    ///
    /// Tolerance reflects the expected global error for each method's order:
    ///   order 1  (Backward Euler):  O(dt)    -> 5 %
    ///   order 2  (CN, SDIRK22/32):  O(dt^2)  -> 0.5 %
    ///   order 3  (SDIRK33):         O(dt^3)  -> 0.05 %
    ///
    /// RK4 global error is O(dt^4) ≈ 1e-8 at t=0.1, negligible as reference.
    #[test]
    fn test_implicit_methods_vs_rk4() {
        let y_rk4 = rk4_reference();
        println!("RK4 reference at t={T_END}: {:?}", y_rk4);

        // (method name, ImplicitBT, expected order, relative tolerance)
        let methods: &[(&str, ImplicitBT, usize, f64)] = &[
            ("ImplicitEuler",   ImplicitBT::implicit_euler(),    1, 5e-2),
            ("CrankNicolson",   ImplicitBT::crank_nicolson(),    2, 5e-3),
            ("SDIRK22",         ImplicitBT::sdirk22(),           2, 5e-3),
            ("SDIRK32",         ImplicitBT::sdirk32(),           2, 5e-3),
            ("SDIRK32_Norsett", ImplicitBT::sdirk32_norsett(),   2, 5e-3),
            ("SDIRK33",         ImplicitBT::sdirk33(),           3, 5e-4),
        ];

        for (name, bt, order, tol_rel) in methods {
            let sys = TestLvSys::new();
            let y0  = faer::mat![[5.0_f64,], [4.0_f64,]];
            let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), bt.clone(), 1e-12, 1e-12);
            run_steps(&mut solver, &sys, DT, N_STEPS);
            let y = solver.state();

            // Compute relative error vs RK4 for reporting
            let rel_err: Vec<f64> = (0..y_rk4.nrows())
                .map(|r| {
                    let diff  = (y[(r, 0)] - y_rk4[(r, 0)]).abs();
                    let scale = y_rk4[(r, 0)].abs().max(1e-8);
                    diff / scale
                })
                .collect();
            println!(
                "{name} (order {order}): y={:?}  rel-err={:?}  tol={tol_rel:.2e}",
                (0..y.nrows()).map(|r| format!("{:.6}", y[(r,0)])).collect::<Vec<_>>(),
                rel_err.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>()
            );

            assert_close_to_rk4(name, &y, &y_rk4, *tol_rel);
        }
    }
}
