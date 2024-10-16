/// Backward differentiation integrators
use faer::prelude::*;
use crate::ode_sys::*;
use crate::newton::*;
use faer::linop::LinOp;
use std::marker::PhantomData;
use faer_gmres::gmres;
use std::collections::VecDeque;


pub struct BdfIntegrator<'a, F>
where
    F: OdeSys<'a>,
{
    /// Order
    order: usize,

    /// System
    sys: &'a F,

    /// Current time
    t: f64,

    /// Storage for past system solution states
    y_hist: VecDeque<Mat<f64>>,

    /// tolerences and newton iteration limits
    // lin_tol: f64,
    // lin_iters: usize,
    // nlin_tol: f64,
    // nlin_iters: usize,

    /// Use a lifetime
    phantom: PhantomData<&'a ()>
}

impl <'a, F> BdfIntegrator <'a, F>
where
    F: OdeSys<'a>,
{
    /// Set the initial conditions and seteup bdf integrator
    pub fn new(t0: f64, y0: MatRef<f64>, order: usize, sys: &'a F) -> Self {
        let mut y_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        Self {
            order,
            sys,
            t: t0,
            y_hist,
            phantom: Default::default()
        }
    }

    fn _nonlin_gfn(&'a self, t: f64, y: MatRef<f64>, dt: f64, order: usize) -> Mat<f64> {
        // current state
        let y0 = self.y_hist[0].as_ref();
        match order {
            // bdf1
            1 => y.as_ref() - y0.as_ref() -dt*self.sys.frhs(t+dt, y),
            // bdf2
            2 => y.as_ref() - (4./3.)*y0.as_ref() + (1./3.)*self.y_hist[1].as_ref() - (2.0*dt/3.0)*self.sys.frhs(t+dt, y),
            _ => panic!("bad order"),
        }
    }

    fn _nonlin_gfn_jac(&'a self, t: f64, y: MatRef<f64>, dt: f64, order: usize) -> ShiftedLinOp<'_> {
        let gamma = 1.0;
        let scale = match order {
            // bdf1
            1 => -dt,
            // bdf2
            2 => -2.0 * dt / 3.0,
            _ => panic!("bad order"),
        };
        self.sys.fjac_shifted(t+dt, y, scale, Some(gamma))
    }

    /// BDF1
    fn step_order_1(&'a self, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // Construct linearop:  Lop := [gamma + scale*J]
        let gamma = 1.0;
        let scale = -dt;
        // let sys_jac_linop_shifted = self.sys.fjac_shifted(t, y0.as_ref(), scale, Some(gamma));

        // Create nonlinear function for the implicit integration formula
        // objective is to find the zero of this function
        // y_k+1 =  y_k + dt * frhs(y_k+1, t+dt) or
        // -dt*frhs(y_k+1, t+dt) + y_k+1 - y_k = 0
        let gfn: &dyn for<'b> Fn(f64, MatRef<'_, f64>) -> Mat<f64> = &|t, y|
            { -dt*self.sys.frhs(t+dt, y) - y0.as_ref() + y.as_ref() };

        // Create jacobian of gfn
        let gfn_jac = |t: f64, y: MatRef<'_, f64>| -> ShiftedLinOp<'a> {
            self.sys.fjac_shifted(t+dt, y, scale, Some(gamma))
        };

        // solve nonlinear system for new y
        let y_new = jac_newton(
            t+dt, y0, &gfn, &gfn_jac,
            1e-6, 1e-8, 100, 1000)?;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }

    /// BDF2
    fn step_order_2(&'a self, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        let gfn = |t: f64, y: MatRef<'_, f64>| -> Mat<f64>
            { self._nonlin_gfn(t, y, dt, self.order) };

        let gfn_jac = |t: f64, y: MatRef<'_, f64>| -> ShiftedLinOp<'a> {
            self._nonlin_gfn_jac(t, y, dt, self.order)
        };

        let y_new = jac_newton(
            t+dt, y0, &gfn, &gfn_jac,
            1e-6, 1e-8, 100, 1000)?;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }

    /// Crank-Nicholson
    fn step_cn(&'a self, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let gamma = 1.0;
        let scale = -0.5*dt;

        // Create nonlinear function for the implicit integration formula
        // objective is to find the zero of this function
        // y_k+1 =  y_k + 0.5*dt * frhs(y_k+1, t+dt) + 0.5*dt * frhs(y_k, t)
        // -0.5*dt*frhs(y_k+1, t+dt) - 0.5*dt*frhs(y_k, t) + y_k+1 - y_k = 0
        let gfn: &dyn for<'b> Fn(f64, MatRef<'_, f64>) -> Mat<f64> = &|t, y|
            { -dt*0.5*self.sys.frhs(t+dt, y) -dt*0.5*self.sys.frhs(t, y0.as_ref()) - y0.as_ref() + y.as_ref() };

        // Create jacobian of gfn
        let gfn_jac = |t: f64, y: MatRef<'_, f64>| -> ShiftedLinOp<'a> {
            self.sys.fjac_shifted(t+dt, y, scale, Some(gamma))
        };

        // solve nonlinear system for new y, might fail
        let y_new = jac_newton(
            t+dt, y0, &gfn, &gfn_jac,
            1e-6, 1e-8, 100, 1000)?;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }
}

impl <'a, F> IntegrateSys<'a> for BdfIntegrator<'a, F>
where
    F: OdeSys<'a>,
{
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step(&'a self, dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        match self.order {
            1 => self.step_order_1(dt),
            2 => {
                if self.y_hist.len() == 2 {
                    self.step_order_2(dt)
                } else {
                    self.step_order_1(dt)
                }
            },
            // not really 3rd order. TODO: add special crank flag
            3 => self.step_cn(dt),
            _ => panic!("bad order"),
       }
    }

    fn time(&'a self) -> &'a Self::TimeType {
        &self.t
    }

    fn state(&'a self) -> &'a Self::SysStateType {
        &self.y_hist[0]
    }

    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>) {
       self.t = s.t;
       self.y_hist.push_front(s.y);
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.y_hist.push_front(y0.to_owned());
        self.t = t0;
    }
}


#[cfg(test)]
mod test_bdf {
    use assert_approx_eq::assert_approx_eq;
    use faer::assert_matrix_eq;
    use crate::ode_utils::{lv_sys_rhs, lv_sys_jac};
    use crate::ode_test_common::*;

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
        let mut sys_solver = BdfIntegrator::new(0.0, y0.as_ref(), 2, &test_sys);

        // step the solution forward
        let mut t = 0.0;
        let dt = 0.01;
        for _i in 0..10 {
            let y_new = sys_solver.step(dt).unwrap();
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
        let mut sys_solver = BdfIntegrator::new(0.0, y0.as_ref(), 2, &test_sys);

        // step the solution forward
        let mut t = 0.0;
        let dt = 0.01;
        for _i in 0..10 {
            let y_new = sys_solver.step(dt).unwrap();
            print!("t:{:?}, y:{:?}", t, &y_new.y);
            sys_solver.accept_step(y_new);
            t += dt;
        }
    }
}
