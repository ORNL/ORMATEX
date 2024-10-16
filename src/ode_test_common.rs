/// Common structs and methods for testing
use faer::prelude::*;
use crate::ode_sys::*;
use crate::ode_utils::{lv_sys_rhs, lv_sys_jac, bateman_sys_rhs};
use faer::linop::LinOp;
use std::marker::PhantomData;

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
