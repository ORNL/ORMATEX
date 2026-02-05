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
/// Common structs and methods for testing
use faer::prelude::*;
use crate::ode_sys::*;
use crate::ode_utils::{lv_sys_rhs, lv_sys_jac, bateman_sys_rhs};
use faer::matrix_free::LinOp;
use std::marker::PhantomData;
use rand::Rng;

/// System with quadratic rhs for testing
pub struct TestQuadSys <'a> {
    sys_x: Mat<f64>,
    phantom: PhantomData<&'a ()>
}
impl <'a> TestQuadSys <'a> {
    pub fn new(sys_x: Mat<f64>) -> Self {
        Self {
            sys_x,
            phantom: Default::default()
        }
    }
}
impl <'a> OdeSys<'a> for TestQuadSys<'a> {
    // define nonlin fn
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
        // x^2 - 1  has zeros a -1, 1
        x.as_ref() * x.as_ref() - faer::Mat::full(x.nrows(), x.ncols(), 1.0)
    }

    // define jacobian of nonlinear fn
    fn fjac(&'a self,
            t: f64,
            x: MatRef<f64>)
        -> Box<dyn LinOp<f64> + 'a>
    {
        // let my_fd_jac = FdJacLinOp::new(1.0, x.to_owned(), self, 1.0, None);
        // Box::new(my_fd_jac)
        Box::new(get_fd_jac(self, t, x))
    }
}

/// Lotka-volterra system with finite diff jacobian
pub struct TestLvFdSys <'a> {
    phantom: PhantomData<&'a ()>
}
impl <'a> TestLvFdSys <'a> {
    pub fn new() -> Self {
        Self {
            phantom: Default::default()
        }
    }
}
impl <'a> OdeSys<'a> for TestLvFdSys<'a> {
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
        lv_sys_rhs(t, x)
    }
    fn fjac<'b>(&'a self,
                t: f64,
                x: MatRef<'b, f64>)
            -> Box<dyn LinOp<f64> + 'a> {
        Box::new(get_fd_jac(self, t, x))
    }
}

/// Lotka-volterra system with exact jacobian
pub struct TestLvSys <'a> {
    phantom: PhantomData<&'a ()>
}
impl <'a> TestLvSys <'a> {
    pub fn new() -> Self {
        Self {
            phantom: Default::default()
        }
    }
}
impl <'a> OdeSys<'a> for TestLvSys<'a> {
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
        lv_sys_rhs(t, x)
    }
    fn fjac<'b>(&'a self,
                t: f64,
                x: MatRef<'b, f64>)
            -> Box<dyn LinOp<f64> + 'a> {
        Box::new(lv_sys_jac(t, x))
    }
}

/// Bateman system with finite diff jacobian
pub struct TestBatemanFdSys <'a> {
    phantom: PhantomData<&'a ()>
}
impl <'a> TestBatemanFdSys <'a> {
    pub fn new() -> Self {
        Self {
            phantom: Default::default()
        }
    }
}
impl <'a> OdeSys<'a> for TestBatemanFdSys<'a> {
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
        bateman_sys_rhs(t, x)
    }
    fn fjac<'b>(&'a self,
                t: f64,
                x: MatRef<'b, f64>)
            -> Box<dyn LinOp<f64> + 'a> {
        Box::new(get_fd_jac(self, t, x))
    }
}

/// Simple test matrix for matexp tests for real eigs
pub fn gen_test_a() -> (Mat<f64>, Mat<f64>)
{
    // Generate a test 3x3 matrix with pure real eigs
    let test_m = faer::mat![
        [-1.0e-1,  0.0,    0.0],
        [ 1.0e-1, -1.0,  0.0],
        [    0.0,  1.0, -1.0e-3],
        ];
    // Generate a test vector
    let test_v = faer::mat![
        [0.1],
        [0.2],
        [0.01],
        ];
    (test_m, test_v)
}

/// Simple test matrix for matexp routines for complex eigs
pub fn gen_test_b() -> (Mat<f64>, Mat<f64>)
{
    // Generate a test 3x3 matrix with one real eig and
    // conjugate complex eigen pair
    let lambda_c = 1.0;
    let lambda_a = 0.5;
    let lambda_b = 0.1;
    let vs = 10.0;
    let test_m = faer::mat![
        [-lambda_a,    -vs,            0.0],
        [ lambda_a+vs, -lambda_b,      0.0],
        [    0.0,       lambda_b, -lambda_c],
        ];
    // eigs = [-1. +0.j       , -0.3+1.2083046j, -0.3-1.2083046j]:
    // Generate a test vector
    let test_v = faer::mat![
        [0.1],
        [0.2],
        [0.01],
        ];
    (test_m, test_v)
}

/// Larger test matrix for matexp routines
pub fn gen_test_c(n: usize) -> (Mat<f64>, Mat<f64>)
{
    let mut rng = rand::thread_rng();
    let lambda_scale = 1.0;
    let vs = 10.0;
    let mut test_m = faer::Mat::zeros(n, n);
    for i in 0..n {
        let lambda: f64 = rng.gen::<f64>() * lambda_scale;
        test_m[(i, i)] = -lambda;
        if i+1 < n {
            test_m[(i+1, i)] = lambda;
        }
    }
    test_m[(1, 0)] += vs;
    test_m[(0, 1)] -= vs;

    // Generate a test vector
    let test_v = faer::Mat::from_fn(n, 1, |_i, _j| {
            rng.gen::<f64>()
        });
    (test_m, test_v)
}
