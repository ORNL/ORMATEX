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
/// Runge-Kutta explicit integrators
use faer::prelude::*;
use std::collections::VecDeque;
use crate::ode_sys::*;

pub struct BT {
    c: Vec<f64>,
    b: Vec<f64>,
    a: Vec<Vec<f64>>,
}

/// Butcher tableau
pub fn bt_factory(order: usize) -> BT {
    match order {
    // RK4
    4 => BT {
        c: vec![0.0, 0.5, 0.5, 1.0],
        b: vec![1./6., 1./3., 1./3., 1./6.],
        a: vec![
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![0.0, 0.0, 1.0],
            ],
        },
    // RK3
    3 => BT {
        c: vec![0.0, 0.5, 1.0],
        b: vec![1./6., 2./3., 1./6.],
        a: vec![
            vec![0.5, 0.0],
            vec![-1., 2.0],
            ],
        },
    // RK2
    2 => BT {
        c: vec![0.0, 0.5],
        b: vec![0.0, 1.0],
        a: vec![
            vec![0.5,],
            ],
        },
    // RK1
    _ => BT {
        c: vec![0.0,],
        b: vec![1.0,],
        a: vec![
            vec![],
            ],
        },
    }
}

/// Runga-Kutta ode intergrator
pub struct RkIntegrator
{
    /// RHS of the ODE system
    // sys: &'a F,

    /// Order
    order: usize,

    /// Butcher tableau
    bt: BT,

    /// Current time
    t: f64,

    /// Solution history storage
    y_hist: VecDeque<Mat<f64>>,
}

impl RkIntegrator
{
    pub fn new(t0: f64, y0: MatRef<f64>, order: usize) -> Self
    {
    let mut y_hist = VecDeque::with_capacity(order);
    y_hist.push_front(y0.to_owned());
        let bt = match order {
            4 => bt_factory(4),
            3 => bt_factory(3),
            2 => bt_factory(2),
            1 => bt_factory(1),
            _ => panic!("Invalid RK order")
        };
        Self {
            order,
            bt,
            t: t0,
            y_hist,
        }
    }

    pub fn step_rk<'a>(&self, sys: &'a dyn OdeSys<'a>, dt: f64) -> Result<StepResult<f64, Mat<f64>>, StepError> {
        // current state
        let t = self.t;
        let y0 = self.y_hist[0].as_ref();

        let mut k: Vec<Mat<f64>> = vec![];
        k.push(sys.frhs(t, y0.as_ref()));
        for i in 0..self.order-1 {
            let mut y_delta = y0.to_owned();
            for j in 0..i+1 {
                y_delta = y_delta.as_ref() + faer::Scale(dt * self.bt.a[i][j]) * k[j].as_ref();
            }
            let k_i = sys.frhs(t + (dt * self.bt.c[i]), y_delta.as_ref());
            k.push(k_i);
        }
        let mut acc = y0.to_owned();
        for i in 0..self.order {
            acc = acc.as_ref() + faer::Scale(dt * self.bt.b[i]) * k[i].as_ref();
        }
        Ok(StepResult::new(t+dt, dt, acc, None))
    }
}

impl <'a> IntegrateSys<'a> for RkIntegrator
{
    type TimeType = f64;
    type SysStateType = Mat<f64>;

    fn step(&self, sys: &'a dyn OdeSys<'a>, dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError> {
       self.step_rk(sys, dt)
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
    }

    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType) {
        self.y_hist.clear();
        self.y_hist.push_front(y0.to_owned());
        self.t = t0;
    }
}


#[cfg(test)]
mod test_rk {
    use crate::ode_test_common::*;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_rk1() {
        // setup system
        let test_sys = TestLvFdSys::new();

        // initial conds
        let y0 = faer::mat![
            [5.0,], // pred pop
            [4.0,], // prey pop
            ];

        // setup rk ode solver order 1
        let mut sys_solver = RkIntegrator::new(0.0, y0.as_ref(), 1);

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
    fn test_rk2() {
        // setup system
        let test_sys = TestLvFdSys::new();

        // initial conds
        let y0 = faer::mat![
            [5.0,], // pred pop
            [4.0,], // prey pop
            ];

        // setup rk ode solver order 2
        let mut sys_solver = RkIntegrator::new(0.0, y0.as_ref(), 2);

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
    fn test_rk4() {
        // setup system
        let test_sys = TestLvFdSys::new();

        // initial conds
        let y0 = faer::mat![
            [5.0,], // pred pop
            [4.0,], // prey pop
            ];

        // setup rk ode solver order 4
        let mut sys_solver = RkIntegrator::new(0.0, y0.as_ref(), 4);

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
