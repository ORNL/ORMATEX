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
/// Backward differentiation integrators
use faer::prelude::*;
use crate::ode_sys::*;
use crate::newton::*;
use std::marker::PhantomData;
use std::collections::VecDeque;


pub struct BdfIntegrator<'a>
{
    /// Order
    order: usize,

    /// Current time
    t: f64,

    /// Storage for past system solution states
    y_hist: VecDeque<Mat<f64>>,

    /// lin solve tolerence
    tol_lin: f64,
    tol_nlin: f64,
    iters_lin: usize,
    iters_nlin: usize,

    /// Use a lifetime
    phantom: PhantomData<&'a ()>
}

impl <'a> BdfIntegrator <'a>
{
    /// Set the initial conditions and seteup bdf integrator
    pub fn new(t0: f64, y0: MatRef<f64>, order: usize) -> Self {
        let mut y_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        Self {
            order,
            t: t0,
            y_hist,
            tol_lin: 1.0e-12,
            tol_nlin: 1.0e-12,
            iters_lin: 1000,
            iters_nlin: 50,
            phantom: Default::default()
        }
    }

    fn _nonlin_gfn<'b>(&self, sys: &'b dyn OdeSys<'b>, t: f64, y: MatRef<f64>, dt: f64, order: usize) -> Mat<f64> {
        // current state
        let y0 = self.y_hist[0].as_ref();
        match order {
            // bdf1
            1 => y.as_ref() - y0.as_ref() -dt*sys.frhs(t+dt, y),
            // bdf2
            2 => y.as_ref() - (4./3.)*y0.as_ref() + (1./3.)*self.y_hist[1].as_ref() - (2.0*dt/3.0)*sys.frhs(t+dt, y),
            _ => panic!("bad order"),
        }
    }

    fn _nonlin_gfn_jac<'b>(&self, sys: &'b dyn OdeSys<'b>, t: f64, y: MatRef<f64>, dt: f64, order: usize) -> ShiftedLinOp<'b> {
        let gamma = 1.0;
        let scale = match order {
            // bdf1
            1 => -dt,
            // bdf2
            2 => -2.0 * dt / 3.0,
            _ => panic!("bad order"),
        };
        sys.fjac_shifted(t+dt, y, scale, Some(gamma))
    }

    /// BDF1
    fn step_order_1<'b>(&self, sys: &'b dyn OdeSys<'b>, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // Construct linearop:  Lop := [gamma + scale*J]
        let gamma = 1.0;
        let scale = -dt;
        // let sys_jac_linop_shifted = sys.fjac_shifted(t, y0.as_ref(), scale, Some(gamma));

        // Create nonlinear function for the implicit integration formula
        // objective is to find the zero of this function
        // y_k+1 =  y_k + dt * frhs(y_k+1, t+dt) or
        // -dt*frhs(y_k+1, t+dt) + y_k+1 - y_k = 0
        let gfn: &dyn for<'c> Fn(f64, MatRef<'_, f64>) -> Mat<f64> = &|t, y|
            { -dt*sys.frhs(t+dt, y) - y0.as_ref() + y.as_ref() };

        // Create jacobian of gfn
        let gfn_jac = |t: f64, y: MatRef<'_, f64>| -> ShiftedLinOp<'_> {
            sys.fjac_shifted(t+dt, y, scale, Some(gamma))
        };

        // solve nonlinear system for new y
        let y_new = jac_newton(
            t+dt, y0, &gfn, &gfn_jac,
            self.tol_nlin, self.tol_lin, self.iters_nlin, self.iters_lin)?;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }

    /// BDF2
    fn step_order_2<'b>(&self, sys: &'b dyn OdeSys<'b>, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        let gfn = |t: f64, y: MatRef<'_, f64>| -> Mat<f64>
            { self._nonlin_gfn(sys, t, y, dt, self.order) };

        let gfn_jac = |t: f64, y: MatRef<'_, f64>| -> ShiftedLinOp<'_> {
            self._nonlin_gfn_jac(sys, t, y, dt, self.order)
        };

        let y_new = jac_newton(
            t+dt, y0, &gfn, &gfn_jac,
            self.tol_nlin, self.tol_lin, self.iters_nlin, self.iters_lin)?;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }

    /// Crank-Nicholson
    fn step_cn<'b>(&self, sys: &'b dyn OdeSys<'b>, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let gamma = 1.0;
        let scale = -0.5*dt;

        // Create nonlinear function for the implicit integration formula
        // objective is to find the zero of this function
        // y_k+1 =  y_k + 0.5*dt * frhs(y_k+1, t+dt) + 0.5*dt * frhs(y_k, t)
        // -0.5*dt*frhs(y_k+1, t+dt) - 0.5*dt*frhs(y_k, t) + y_k+1 - y_k = 0
        let gfn: &dyn for<'c> Fn(f64, MatRef<'_, f64>) -> Mat<f64> = &|t, y|
            { -dt*0.5*sys.frhs(t+dt, y) -dt*0.5*sys.frhs(t, y0.as_ref()) - y0.as_ref() + y.as_ref() };

        // Create jacobian of gfn
        let gfn_jac = |t: f64, y: MatRef<'_, f64>| -> ShiftedLinOp<'_> {
            sys.fjac_shifted(t+dt, y, scale, Some(gamma))
        };

        // solve nonlinear system for new y, might fail
        let y_new = jac_newton(
            t+dt, y0, &gfn, &gfn_jac,
            self.tol_nlin, self.tol_lin, self.iters_nlin, self.iters_lin)?;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }
}

impl <'a> IntegrateSys<'a> for BdfIntegrator<'a>
{
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(&mut self, sys: &'b dyn OdeSys<'b>, dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        match self.order {
            1 => self.step_order_1(sys, dt),
            2 => {
                if self.y_hist.len() == 2 {
                    self.step_order_2(sys, dt)
                } else {
                    self.step_order_1(sys, dt)
                }
            },
            // not really 3rd order. TODO: add special crank flag
            3 => self.step_cn(sys, dt),
            _ => panic!("bad order"),
       }
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
       if self.y_hist.len() >= self.order+1 {
           self.y_hist.pop_back();
       }
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.y_hist.push_front(y0.to_owned());
        self.t = t0;
    }
}


#[cfg(test)]
mod test_bdf {
    use crate::test_common::*;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_bdf1_jfnk() {
        // test with only access to sys rhs.  jacobian-vector prods by finite diff.
        // full jacobian never is constructed.

        // setup system
        let test_sys = TestLvFdSys::new();

        // initial conds
        let y0 = faer::mat![
            [5.0,], // pred pop
            [4.0,], // prey pop
            ];

        // setup the integrator
        let mut sys_solver = BdfIntegrator::new(0.0, y0.as_ref(), 2);

        // step the solution forward
        let mut t = 0.0;
        let dt = 0.01;
        for _i in 0..10 {
            let y_new = sys_solver.step(&test_sys, dt).unwrap();
            print!("t:{:?}, y:{:?}", t, &y_new.y);
            sys_solver.accept_step(y_new);
            t += dt;
        }
    }

    #[test]
    fn test_bdf1_nk() {
        // test with full exact matrix jacobian
        // setup system
        let test_sys = TestLvSys::new();

        // initial conds
        let y0 = faer::mat![
            [5.0,], // pred pop
            [4.0,], // prey pop
            ];

        // setup the integrator
        let mut sys_solver = BdfIntegrator::new(0.0, y0.as_ref(), 2);

        // step the solution forward
        let mut t = 0.0;
        let dt = 0.01;
        for _i in 0..10 {
            let y_new = sys_solver.step(&test_sys, dt).unwrap();
            print!("t:{:?}, y:{:?}", t, &y_new.y);
            sys_solver.accept_step(y_new);
            t += dt;
        }
    }
}
