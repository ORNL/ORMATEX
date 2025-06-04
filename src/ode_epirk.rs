/// Exponential prop-iterative RK class of exponential integrators
///
use faer::prelude::*;
use num_traits::real::Real;
use num_traits::Float;
use crate::matexp_krylov::KrylovExpm;
use crate::ode_sys::*;
use faer::matrix_free::LinOp;
use faer::dyn_stack::PodStack;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
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

    /// tol used to check max derivative for nonautonomous system
    tol_fdt: f64,

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
            tol_fdt: -1.0,
            y_hist,
            t_hist,
            phantom: Default::default()
        }
    }

    /// builder fn to set optional solver parameters
    pub fn with_opt(mut self, option_str: String, option_val: f64) -> Self
    {
        match option_str.as_str() {
            "tol_fdt" => { self.tol_fdt = option_val },
            _ => panic!("bad option")
        };
        self
    }

    /// Computes remainder R(yr) = frhs(yr) - frhs(y0) - J_y0*(yr-y0) - v*t
    /// where if v=d(Frhs)/dt is nonzero for nonautonomous systems
    fn remf(&self, tr: f64, yr: MatRef<f64>, frhs_y0: MatRef<f64>, sys_jac_lop_y0: &dyn LinOp<f64>, v: Option<MatRef<f64>>)
        -> Mat<f64>
    {
        let t = self.t_hist[0];
        let y0 = self.y_hist[0].as_ref();
        let frhs_yr = self.sys.frhs(tr, yr);

        let mut jac_yd = faer::Mat::zeros(y0.nrows(), 1);
        sys_jac_lop_y0.apply(
            jac_yd.as_mut(),
            (yr.as_ref()-y0.as_ref()).as_ref(),
            faer::get_global_parallelism(),
            MemStack::new(&mut MemBuffer::new(StackReq::empty()))
        );

        let dt = tr - t;
        let vn_t = Scale(dt) * v.unwrap_or(Mat::zeros(yr.nrows(), yr.ncols()).as_ref());
        frhs_yr - frhs_y0 - jac_yd - vn_t
    }

    /// Estimates the time drivative of frhs by finite difference
    fn frhs_fdt(&self, fy0: MatRef<f64>, del_t: f64) -> Mat<f64> {
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let fy1 = self.sys.frhs(t+del_t, y0);
        (fy1 - fy0) / Scale(del_t)
    }

    /// Correction for nonautonomous case
    fn fphi2_v(&self, fy0: MatRef<f64>, sys_jac_lop: &dyn LinOp<f64>, dt: f64) -> (Mat<f64>, Mat<f64>) {
        let mut phi2_v = Mat::zeros(fy0.nrows(), fy0.ncols());
        if self.tol_fdt < 0. {
            return (phi2_v,  Mat::zeros(fy0.nrows(), fy0.ncols()))
        }
        let v = self.frhs_fdt(fy0.as_ref(), 1e-8);
        if v.norm_max() > self.tol_fdt {
            phi2_v = Scale(dt.powi(2)) * self.expm.apply_phi_linop(sys_jac_lop, dt, v.as_ref(), 2);
        }
        (phi2_v, v)
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

        // correction for nonautonomous case
        let (phi2_v, _) = self.fphi2_v(fy0.as_ref(), sys_jac_lop.as_ref(), dt);

        let y_new = y0.as_ref() + phi2_v +
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

        // correction for nonautonomous case
        let (phi2_v, v) = self.fphi2_v(fy0.as_ref(), sys_jac_lop.as_ref(), dt);

        let t_2 = t + dt;
        let y_2 = y0.as_ref() + phi2_v.as_ref() +
            self.expm.apply_phi_linop(
                sys_jac_lop.as_ref(),
                dt, fy0_dt.as_ref(), 1);
        // remainder fn
        let r_2 = self.remf(
            t_2, y_2.as_ref(), fy0.as_ref(), sys_jac_lop.as_ref(), Some(v.as_ref()));

        // compute final update
        let y_new = y_2.as_ref() + 2.*dt*self.expm.apply_phi_linop(
            sys_jac_lop.as_ref(), dt, r_2.as_ref(), 3);

        // err est
        let y_err = (y_new.as_ref() - y_2.as_ref()).as_ref().norm_l1().abs();

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

        // correction for nonautonomous case
        let (phi2_v, v) = self.fphi2_v(fy0.as_ref(), sys_jac_lop.as_ref(), dt);

        let rn_dt = faer::Scale(dt * 2.0 / 3.0) * self.remf(
            tp, yp.as_ref(), fy0.as_ref(), sys_jac_lop.as_ref(), Some(v.as_ref()));

        // only need single apply linop using kiops
        // build vector of rhs
        let zero_mat = faer::Mat::zeros(y0.nrows(), 1);
        let vb = vec![
            zero_mat.as_ref(),
            fy0_dt.as_ref(),
            rn_dt.as_ref(),
        ];
        let ext_a_lo = ExtendedLinOp::new(dt, sys_jac_lop, &vb);
        let y_new = y0.as_ref() + self.expm.kiops_fixedsteps(&ext_a_lo, 1.0, &vb) + phi2_v;

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
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
