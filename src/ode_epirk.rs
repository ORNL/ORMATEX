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
/// Exponential prop-iterative RK class of exponential integrators
///
use faer::prelude::*;
use crate::matexp_traits::LinOpPhikvEvaluator;
use crate::ode_sys::*;
use faer::matrix_free::LinOp;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use std::collections::VecDeque;


pub struct EpirkIntegrator<T: LinOpPhikvEvaluator>
{
    /// Matrix exponential evaluator
    expm: T,

    /// Order
    order: usize,

    /// Method
    method: String,

    /// Current time
    t: f64,

    /// tol used to check max derivative for nonautonomous system
    tol_fdt: f64,

    /// Storage for past system solution states
    y_hist: VecDeque<Mat<f64>>,
    t_hist: VecDeque<f64>,
}

impl <T> EpirkIntegrator <T>
where
    T: LinOpPhikvEvaluator
{
    /// Set the initial conditions and seteup bdf integrator
    pub fn new(t0: f64, y0: MatRef<f64>, method: String, expm: T) -> Self
    {
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
            t: t0,
            tol_fdt: -1.0,
            y_hist,
            t_hist,
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
    fn remf<'b>(&self, sys: &'b dyn OdeSys<'b>, tr: f64, yr: MatRef<f64>, frhs_y0: MatRef<f64>, sys_jac_lop_y0: &dyn LinOp<f64>, v: Option<MatRef<f64>>)
        -> Mat<f64>
    {
        let t = self.t_hist[0];
        let y0 = self.y_hist[0].as_ref();
        let frhs_yr = sys.frhs(tr, yr);

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
    fn frhs_fdt<'b>(&self, sys: &'b dyn OdeSys<'b>, fy0: MatRef<f64>, del_t: f64) -> Mat<f64> {
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let fy1 = sys.frhs(t+del_t, y0);
        (fy1 - fy0) / Scale(del_t)
    }

    /// Correction for nonautonomous case
    fn fphi2_v<'b>(&self, sys: &'b dyn OdeSys<'b>, fy0: MatRef<f64>, sys_jac_lop: &dyn LinOp<f64>, dt: f64) -> (Mat<f64>, Mat<f64>) {
        let mut phi2_v = Mat::zeros(fy0.nrows(), fy0.ncols());
        if self.tol_fdt < 0. {
            return (phi2_v,  Mat::zeros(fy0.nrows(), fy0.ncols()))
        }
        let v = self.frhs_fdt(sys, fy0.as_ref(), 1e-8);
        if v.norm_max() > self.tol_fdt {
            phi2_v = Scale(dt.powi(2)) * self.expm.apply_phi_k(sys_jac_lop, dt, v.as_ref(), 2);
        }
        (phi2_v, v)
    }

    /// Exponential Propagative Iterative Order 2 method (EPI3)
    ///
    /// Gaudreault, Stéphane, and Janusz A. Pudykiewicz.
    /// An efficient exponential time integration method for the numerical
    /// solution of the shallow water equations on the sphere.
    /// Journal of Computational Physics 322 (2016): 827-848.
    ///
    /// Tokman, Mayya. Efficient integration of large stiff systems of ODEs
    /// with exponential propagation iterative (EPI) methods.
    /// Journal of Computational Physics 213.2 (2006): 748-776.
    /// EPI2
    fn step_order_2<'b>(&mut self, sys: &'b dyn OdeSys<'b>, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // setup jacobian linear operator evaluated at y0
        let sys_jac_lop = sys.fjac(t, y0.as_ref());
        let fy0 = sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);

        // correction for nonautonomous case
        let v: Mat<f64> = if self.tol_fdt < 0.0 {
                faer::Mat::zeros(y0.nrows(), 1)
            } else {
                self.frhs_fdt(sys, fy0.as_ref(), 1e-8)
            };
        let vb2 = dt.powi(2) * v;

        // build vector of rhs
        let zero_mat = faer::Mat::zeros(y0.nrows(), 1);
        let vb = vec![
            zero_mat.as_ref(),
            fy0_dt.as_ref(),
            vb2.as_ref(),
        ];
        let ext_a_lo = DynRefExtendedLinOp::new(dt, sys_jac_lop.as_ref(), &vb);
        self.expm.apply_prepare(sys_jac_lop.as_ref(), dt, y0.as_ref(), 2);
        let y_new = y0.as_ref() + self.expm.apply_phi_k_v(&ext_a_lo, 1.0, &vb);

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }

    /// EXPRB32
    /// Exponential Rosenroack order 3 with 2nd order embedded error estimate.
    /// Ref: Hochbruck, Marlis, Alexander Ostermann, and Julia Schweitzer.
    /// Exponential Rosenbrock-type methods.
    /// SIAM Journal on Numerical Analysis 47.1 (2009): 786-803.
    fn step_exprb32<'b>(&mut self, sys: &'b dyn OdeSys<'b>, dt: f64)
        -> Result<StepResult<f64, Mat<f64>>, StepError>
    {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        // setup jacobian linear operator evaluated at y0
        let sys_jac_lop = sys.fjac(t, y0.as_ref());
        let fy0 = sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);
        self.expm.apply_prepare(sys_jac_lop.as_ref(), dt, y0.as_ref(), 3);

        // correction for nonautonomous case
        let (phi2_v, v) = self.fphi2_v(sys, fy0.as_ref(), sys_jac_lop.as_ref(), dt);

        let t_2 = t + dt;
        let y_2 = y0.as_ref() + phi2_v.as_ref() +
            self.expm.apply_phi_k(
                sys_jac_lop.as_ref(),
                dt, fy0_dt.as_ref(), 1);
        // remainder fn
        let r_2 = self.remf(
            sys, t_2, y_2.as_ref(), fy0.as_ref(), sys_jac_lop.as_ref(), Some(v.as_ref()));

        // compute final update
        let y_new = y_2.as_ref() + 2.*dt*self.expm.apply_phi_k(
            sys_jac_lop.as_ref(), dt, r_2.as_ref(), 3);

        // err est
        let y_err = (y_new.as_ref() - y_2.as_ref()).as_ref().norm_l1().abs();

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, Some(y_err)))
    }

    /// Exponential Propagative Iterative Order 3 method (EPI3)
    ///
    /// Gaudreault, Stéphane, and Janusz A. Pudykiewicz.
    /// An efficient exponential time integration method for the numerical
    /// solution of the shallow water equations on the sphere.
    /// Journal of Computational Physics 322 (2016): 827-848.
    /// solution of the shallow water equations.
    fn step_order_3<'b>(&mut self, sys: &'b dyn OdeSys<'b>, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();
        let yp = self.y_hist[1].as_ref();
        let tp = self.t_hist[1];

        let sys_jac_lop = sys.fjac(t, y0.as_ref());
        let fy0 = sys.frhs(t, y0);
        let fy0_dt = fy0.as_ref() * faer::Scale(dt);

        // correction for nonautonomous case
        let v: Mat<f64> = if self.tol_fdt < 0.0 {
                faer::Mat::zeros(y0.nrows(), 1)
            } else {
                self.frhs_fdt(sys, fy0.as_ref(), 1e-8)
            };

        let rn_dt = faer::Scale(dt * 2.0 / 3.0) * self.remf(
            sys, tp, yp.as_ref(), fy0.as_ref(), sys_jac_lop.as_ref(), Some(v.as_ref()));
        let vb2 = rn_dt + dt.powi(2) * v;

        // build vector of rhs
        let zero_mat = faer::Mat::zeros(y0.nrows(), 1);
        let vb = vec![
            zero_mat.as_ref(),
            fy0_dt.as_ref(),
            vb2.as_ref(),
        ];
        let ext_a_lo = DynRefExtendedLinOp::new(dt, sys_jac_lop.as_ref(), &vb);
        self.expm.apply_prepare(sys_jac_lop.as_ref(), dt, y0.as_ref(), 2);
        let y_new = y0.as_ref() + self.expm.apply_phi_k_v(&ext_a_lo, 1.0, &vb);

        // return result
        Ok(StepResult::new(t+dt, dt, y_new, None))
    }
}

impl <'a, T> IntegrateSys<'a> for EpirkIntegrator<T>
where
    T: LinOpPhikvEvaluator
{
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step<'b>(&mut self, sys: &'b dyn OdeSys<'b>,  dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
        println!("\nEPI step, t: {:?}, dt: {:?}", self.t, dt);
        let clock = std::time::Instant::now();
        let res = match self.method.as_str() {
            "epi2" | "exprb2" => {
                self.step_order_2(sys, dt)
            },
            "epi3" => {
                if self.y_hist.len() >= 2 {
                    self.step_order_3(sys, dt)
                } else {
                    self.step_order_2(sys, dt)
                }
            },
            "exprb3" => {
                self.step_exprb32(sys, dt)
            },
            _ => panic!("bad method"),
        };
        println!("EPI step time (s): {}", clock.elapsed().as_secs_f64());
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
       self.t_hist.push_front(s.t);
       if self.y_hist.len() >= self.order+1 {
           self.y_hist.pop_back();
           self.t_hist.pop_back();
       }
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.t_hist.clear();
        self.y_hist.push_front(y0.to_owned());
        self.t_hist.push_front(t0);
        self.t = t0;
    }

}
