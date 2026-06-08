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
/// BDF1 and Crank–Nicolson are tableau-expressible and delegate to
/// `dirk_step`/`DirkIntegrator`.  BDF2 is a multistep method that keeps its
/// own history-based implementation.

use faer::prelude::*;
use crate::ode_sys::*;
use crate::newton::*;
use crate::tableau_implicit::ImplicitBT;
use std::marker::PhantomData;
use std::collections::VecDeque;


// ─── Generic DIRK step (free function) ───────────────────────────────────────

/// Advance `y' = f(t,y)` by one step `dt` using the implicit Butcher tableau `bt`.
///
/// For each stage `i`:
///
/// 1. Build the explicit accumulation
///    `y_expl = y0 + dt · Σ_{j < i} a[i][j] · k[j]`
///
/// 2. **Explicit stage** (`a[i][i] == 0`):
///    `k[i] = f(t + c[i]·dt, y_expl)` — no solve needed.
///
/// 3. **Implicit stage** (`a[i][i] != 0`):
///    Solve  `g(y_i) = y_i − y_expl − dt·a[i][i]·f(t_i, y_i) = 0`
///    using Newton–Krylov, starting from `y_expl`, with Jacobian
///    `I − dt·a[i][i]·J_f ≡ fjac_shifted(t_i, y_i, −dt·a[i][i], γ=1)`.
///    Then `k[i] = f(t_i, y_i)`.
///
/// Finally `y_{n+1} = y0 + dt · Σ_i b[i] · k[i]`.
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
        // ── explicit accumulation: y_expl = y0 + dt * Σ_{j<i} a[i][j]*k[j] ──
        let mut y_expl: Mat<f64> = y0.to_owned();
        for j in 0..i {
            y_expl = y_expl.as_ref()
                + faer::Scale(dt * bt.a[i][j]) * k[j].as_ref();
        }

        let a_ii = bt.a[i][i];
        let t_i  = t + bt.c[i] * dt;

        let k_i: Mat<f64> = if a_ii == 0.0 {
            // ── Explicit stage ───────────────────────────────────────────────
            sys.frhs(t_i, y_expl.as_ref())

        } else {
            // ── Implicit stage ───────────────────────────────────────────────
            // Solve  g(y_i) = y_i - y_expl - dt*a_ii*f(t_i, y_i) = 0
            // ∂g/∂y_i = I - dt*a_ii * J_f(t_i, y_i)
            //         ≡ fjac_shifted(t_i, y_i, scale=-dt*a_ii, gamma=1.0)
            //
            // The `gfn` closure captures `y_expl` by shared ref; `gfn_jac` is
            // annotated to return `ShiftedLinOp<'jac>` so the lifetime of
            // `y_expl` (used only as initial guess) stays decoupled from `'jac`.
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
            // Its lifetime is local — safe because jac_newton clones it immediately
            // (see the decoupled 'jac / '_ lifetimes in newton::jac_newton).
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

    // ── final accumulation: y_{n+1} = y0 + dt * Σ_i b[i]*k[i] ──────────────
    let mut y_new: Mat<f64> = y0.to_owned();
    for i in 0..s {
        y_new = y_new.as_ref() + faer::Scale(dt * bt.b[i]) * k[i].as_ref();
    }
    Ok(StepResult::new(t + dt, dt, y_new, None))
}


// ─── DirkIntegrator ──────────────────────────────────────────────────────────

/// Generic single-step DIRK / SDIRK integrator driven by an [`ImplicitBT`] tableau.
///
/// Works with any fully-implicit or ESDIRK tableau: Backward Euler,
/// Crank–Nicolson, SDIRK22, SDIRK32, SDIRK32 (Norsett), SDIRK33, …
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
    pub fn new(t0: f64, y0: MatRef<'_, f64>, bt: ImplicitBT) -> Self {
        Self {
            bt,
            t: t0,
            y: y0.to_owned(),
            tol_lin:    1.0e-12,
            tol_nlin:   1.0e-12,
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


// ─── BdfIntegrator ───────────────────────────────────────────────────────────

/// BDF linear multistep integrator.
///
/// `order = 1` — BDF1 (Backward Euler); delegates to `dirk_step` with
///               `ImplicitBT::implicit_euler()`.
///
/// `order = 2` — BDF2; requires solution history. Bootstraps with BDF1 on the
///               first step when history is not yet full.  Cannot be expressed
///               as a Butcher tableau (multistep method).
///
/// `order = 3` — Crank–Nicolson (legacy alias); delegates to `dirk_step` with
///               `ImplicitBT::crank_nicolson()`.
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
    pub fn new(t0: f64, y0: MatRef<'_, f64>, order: usize) -> Self {
        let mut y_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        Self {
            order,
            t: t0,
            y_hist,
            tol_lin:    1.0e-12,
            tol_nlin:   1.0e-12,
            iters_lin:  1000,
            iters_nlin: 50,
            phantom: Default::default(),
        }
    }

    // ── BDF1 → delegates to ImplicitBT::implicit_euler() ────────────────────

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

    // ── BDF2: linear multistep — history-based, no Butcher tableau ──────────

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
        //   ∂g/∂y = I − (2/3)*dt*J_f  ≡  fjac_shifted(scale=−2dt/3, gamma=1)
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


// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod test_bdf {
    use crate::test_common::*;
    use super::*;

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

    // ── BDF legacy tests (unchanged behaviour) ───────────────────────────────

    #[test]
    fn test_bdf1_jfnk() {
        let sys = TestLvFdSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = BdfIntegrator::new(0.0, y0.as_ref(), 2);
        let dt = 0.01;
        for _ in 0..10 {
            let res = solver.step(&sys, dt).unwrap();
            print!("t:{:?}, y:{:?}", solver.time(), &res.y);
            solver.accept_step(res);
        }
    }

    #[test]
    fn test_bdf1_nk() {
        let sys = TestLvSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = BdfIntegrator::new(0.0, y0.as_ref(), 2);
        let dt = 0.01;
        for _ in 0..10 {
            let res = solver.step(&sys, dt).unwrap();
            print!("t:{:?}, y:{:?}", solver.time(), &res.y);
            solver.accept_step(res);
        }
    }

    // ── DirkIntegrator tests ─────────────────────────────────────────────────

    #[test]
    fn test_sdirk22_fd() {
        let sys = TestLvFdSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk22());
        run_steps(&mut solver, &sys, 0.01, 10);
        println!("SDIRK22 FD:  y = {:?}", solver.state());
    }

    #[test]
    fn test_sdirk22_exact_jac() {
        let sys = TestLvSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk22());
        run_steps(&mut solver, &sys, 0.01, 10);
        println!("SDIRK22 exact Jac: y = {:?}", solver.state());
    }

    #[test]
    fn test_sdirk32_fd() {
        let sys = TestLvFdSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32());
        run_steps(&mut solver, &sys, 0.01, 10);
        println!("SDIRK32 FD:  y = {:?}", solver.state());
    }

    #[test]
    fn test_sdirk32_exact_jac() {
        let sys = TestLvSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32());
        run_steps(&mut solver, &sys, 0.01, 10);
        println!("SDIRK32 exact Jac: y = {:?}", solver.state());
    }

    #[test]
    fn test_sdirk32_norsett_exact_jac() {
        let sys = TestLvSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk32_norsett());
        run_steps(&mut solver, &sys, 0.01, 10);
        println!("SDIRK32 Norsett: y = {:?}", solver.state());
    }

    #[test]
    fn test_sdirk33_exact_jac() {
        let sys = TestLvSys::new();
        let y0  = faer::mat![[5.0,], [4.0,]];
        let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::sdirk33());
        run_steps(&mut solver, &sys, 0.01, 10);
        println!("SDIRK33 Alexander: y = {:?}", solver.state());
    }

    /// All DIRK variants should produce results close to BDF1 on Lotka–Volterra
    /// with a small step size (dt = 0.001).  We accept 1 % relative agreement.
    #[test]
    fn test_dirk_methods_agree_with_bdf1() {
        let y0 = faer::mat![[5.0,], [4.0,]];
        let dt = 0.001;
        let n  = 10;

        // BDF1 baseline
        let ref_sys = TestLvSys::new();
        let mut ref_solver = BdfIntegrator::new(0.0, y0.as_ref(), 1);
        run_steps(&mut ref_solver, &ref_sys, dt, n);
        let y_ref = ref_solver.state();

        let methods: &[(&str, ImplicitBT)] = &[
            ("ImplicitEuler",  ImplicitBT::implicit_euler()),
            ("CrankNicolson",  ImplicitBT::crank_nicolson()),
            ("SDIRK22",        ImplicitBT::sdirk22()),
            ("SDIRK32",        ImplicitBT::sdirk32()),
            ("SDIRK32_Norsett",ImplicitBT::sdirk32_norsett()),
            ("SDIRK33",        ImplicitBT::sdirk33()),
        ];

        for (name, bt) in methods {
            let sys = TestLvSys::new();
            let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), bt.clone());
            run_steps(&mut solver, &sys, dt, n);
            let y = solver.state();
            println!("{name}: y = {:?}", y);
            for row in 0..y_ref.nrows() {
                let tol = 0.01 * y_ref[(row, 0)].abs().max(1e-10);
                let diff = (y[(row, 0)] - y_ref[(row, 0)]).abs();
                assert!(
                    diff < tol,
                    "{name} row {row}: got {}, expected {}, diff {diff} > tol {tol}",
                    y[(row, 0)], y_ref[(row, 0)]
                );
            }
        }
    }
}
