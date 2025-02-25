/// Exponential prop-iterative RK class of exponential integrators
///
use faer::prelude::*;
use crate::matexp_krylov::KrylovExpm;
use crate::ode_sys::*;
use faer::matrix_free::LinOp;
use faer::dyn_stack::PodStack;
use std::marker::PhantomData;
use std::collections::VecDeque;


pub struct EpirkIntegrator<'a, F>
where
    F: OdeSys<'a>,
{
    /// Matrix exponential evaluator
    expm: KrylovExpm,

    /// Order
    order: usize,

    /// Method
    method: String,

    /// System
    sys: &'a F,

    /// Current time
    t: f64,

    /// Storage for past system solution states
    y_hist: VecDeque<Mat<f64>>,
    t_hist: VecDeque<f64>,

    /// Use a lifetime
    phantom: PhantomData<&'a ()>
}

impl <'a, F> EpirkIntegrator <'a, F>
where
    F: OdeSys<'a>,
{
    /// Set the initial conditions and seteup bdf integrator
    pub fn new(t0: f64, y0: MatRef<f64>, method: String, sys: &'a F, expm: KrylovExpm) -> Self {
        let order = match method.as_str() {
            "epi2" | "exprb2" => 2,
            "epi3" | "exprb3" => 3,
            _ => panic!("invalid method: {:?}. Valid: epi2,epi3,exprb2,exprb3", method),
        };
        let mut y_hist = VecDeque::with_capacity(order);
        let mut t_hist = VecDeque::with_capacity(order);
        y_hist.push_front(y0.to_owned());
        t_hist.push_front(t0);
        Self {
            expm,
            order,
            method,
            sys,
            t: t0,
            y_hist,
            t_hist,
            phantom: Default::default()
        }
    }

    /// Computes remainder R(yr) = frhs(yr) - frhs(y0) - J_y0*(yr-y0)
    fn remf(&self, tr: f64, yr: MatRef<f64>, frhs_y0: MatRef<f64>, sys_jac_lop_y0: &dyn LinOp<f64>) -> Mat<f64> {
        let y0 = self.y_hist[0].as_ref();
        let frhs_yr = self.sys.frhs(tr, yr);

        let mut _dummy_podstack: [u8;1] = [0u8;1];
        let mut jac_yd = faer::Mat::zeros(y0.nrows(), 1);
        sys_jac_lop_y0.apply(jac_yd.as_mut(),
                          (yr.as_ref()-y0.as_ref()).as_ref(),
                          faer::get_global_parallelism(),
                          PodStack::new(&mut _dummy_podstack));

        frhs_yr - frhs_y0 - jac_yd
    }

    /// EPI2
    fn step_order_2(&self, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // setup jacobian linear operator evaluated at y0
        let sys_jac_lop = self.sys.fjac(t, y0.as_ref());
        let fy0 = self.sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);
        let y_new = y0.as_ref() +
            self.expm.apply_phi_linop(
                sys_jac_lop.as_ref(),
                dt, fy0_dt.as_ref(), 1);

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }

    /// EXPRB32
    /// Exponential Rosenroack order 3 with 2nd order embedded error estimate.
    fn step_exprb32(&self, dt: f64)
        -> Result<StepResult<f64, Mat<f64>>, StepError>
    {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // setup jacobian linear operator evaluated at y0
        let sys_jac_lop = self.sys.fjac(t, y0.as_ref());
        let fy0 = self.sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);
        let t_2 = t + dt;
        let y_2 = y0.as_ref() +
            self.expm.apply_phi_linop(
                sys_jac_lop.as_ref(),
                dt, fy0_dt.as_ref(), 1);
        // remainder fn
        let r_2 = self.remf(
            t_2, y_2.as_ref(), fy0.as_ref(), sys_jac_lop.as_ref());

        // compute final update
        let y_new = y_2.as_ref() + 2.*dt*self.expm.apply_phi_linop(
            sys_jac_lop.as_ref(), dt, r_2.as_ref(), 3);

        // err est
        let y_err = (y_new.as_ref() - y_2.as_ref()).norm_l1().abs();

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, Some(y_err)))
    }

    /// EPI3
    /// From Gaudreault et. al.
    /// An efficient exponential time integration method for the numerical
    /// solution of the shallow water equations.
    fn step_order_3(&self, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let yp = self.y_hist[1].as_ref();
        let tp = self.t_hist[1];

        let sys_jac_lop = self.sys.fjac(t, y0.as_ref());

        let fy0 = self.sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);
        let y1 = y0.as_ref() +
            self.expm.apply_phi_linop(sys_jac_lop.as_ref(), dt, fy0_dt.as_ref(), 1);

        let rn_dt = self.remf(tp, yp.as_ref(), fy0.as_ref(), sys_jac_lop.as_ref()) * faer::Scale(dt);
        let y_new = y1.as_ref() + faer::Scale(2.0/3.0)*
            self.expm.apply_phi_linop(sys_jac_lop.as_ref(), dt, rn_dt.as_ref(), 2);

        // estimate error in the step
        let y_err = (y_new.as_ref() - y1.as_ref()).norm_l1().abs();

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, Some(y_err)))
    }
}

impl <'a, F> IntegrateSys<'a> for EpirkIntegrator<'a, F>
where
    F: OdeSys<'a>,
{
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step(&self, dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        match self.method.as_str() {
            "epi2" | "exprb2" => {
                self.step_order_2(dt)
            },
            "epi3" => {
                if self.y_hist.len() >= 2 {
                    self.step_order_3(dt)
                } else {
                    self.step_order_2(dt)
                }
            },
            "exprb3" => {
                self.step_exprb32(dt)
            },
            _ => panic!("bad method"),
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
       self.t_hist.push_front(s.t);
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.y_hist.push_front(y0.to_owned());
        self.t_hist.push_front(t0);
        self.t = t0;
    }

}
