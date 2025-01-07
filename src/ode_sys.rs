/// Defines and ODE system of equations
/// Defines interface for integration ode equations with
/// exponential integrators, implicit and explicit integrators
///
use faer::prelude::*;
use faer::linop::LinOp;
use faer::Parallelism;
use faer::dyn_stack::PodStack;
use std::ops::{Add, Sub};
use std::{error::Error, fmt};

#[derive(Debug)]
pub struct StepError
{
    pub error_code: usize,
    pub msg: String,
}

impl Error for StepError
{}

impl fmt::Display for StepError
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StepError")
    }
}


pub struct StepResult<T, S> {
    // Current system time
    pub t: T,
    // Time step size
    pub dt: T,
    // Current system state
    pub y: S,
    // Not-None if embeded method provides err estimate
    pub err: Option<f64>,
}
impl <T, S> StepResult<T, S> {
    pub fn new(t: T, dt: T, y: S, err: Option<f64>) -> Self
    {
        Self {
            t,
            dt,
            y,
            err,
        }
    }
}

pub trait IntegrateSys <'a>
{
    type TimeType;
    type SysStateType;

    /// Step solution forward by dt, proposes a new state.
    /// This may outright fail due to numerical issue
    fn step(&'a self, dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError>;

    /// Get current time
    fn time(&'a self) -> &'a Self::TimeType;

    /// Get current system state
    fn state(&'a self) -> &'a Self::SysStateType;

    /// Accepts the proposed new time and state.
    /// Records accepted state into solution history.
    fn accept_step(&mut self, s: StepResult<Self::TimeType, Self::SysStateType>);

    /// Reset integrator.  Removes solution history
    fn reset_ic(&mut self, t0: Self::TimeType, y0: Self::SysStateType);

//     fn int(&'a mut self, y0: MatRef<f64>, mut y: MatMut<f64>, ti: f64, tf: f64,  dt_max: f64)
//     {
//         y.copy_from(y0);
//         let mut t = ti.clone();
//         let mut dt = f64::min(dt_max, tf - ti);
//         let mut i = 0;
//         loop {
//             let leftover = tf - t;
//             dt = f64::min(leftover, dt);
//             if t < tf {
//                 y.copy_from(self.step(dt));
//                 t += dt;
//             }
//             else {
//                 break;
//             }
//             i += 1;
//         }
//     }
}


/// Helper method to apply the linop to a vec but does an extra allocation to store
/// and return the result.
pub fn apply_linop(lop: &impl LinOp<f64>, q: MatRef<f64>) -> Mat<f64> {
    let mut out = faer::Mat::zeros(lop.nrows(), q.ncols());
    let mut _dummy_podstack: [u8;1] = [0u8;1];
    lop.apply(
        out.as_mut(),
        q,
        faer::get_global_parallelism(),
        PodStack::new(&mut _dummy_podstack)
        );
    out
}


/// Wrapper to shift a LinOp, A
/// and applies
/// [[ A,  B],
///  [ 0,  K]]
/// to a vector.
/// Note: block matricies can be built in faer with
/// mat.get_mut(1..5,3..6).copy_from(other.get(0..4, 0..3))
pub struct ExtendedLinOp<'a> {
    t: f64,
    inner_lop: Box<dyn LinOp<f64> + 'a>,
    B: faer::Mat<f64>,
    K: faer::Mat<f64>,
}


/// Wrapper to shift and scale a LinOp
pub struct ShiftedLinOp<'a> {
    t: f64,
    inner_lop: Box<dyn LinOp<f64> + 'a>,
    scale: f64,
    gamma: Option<f64>,
}

impl <'a> ShiftedLinOp <'a> {
    pub fn new(t: f64, inner_lop: Box<dyn LinOp<f64> + 'a>, scale: f64, gamma: Option<f64>)
    -> Self {
        Self {
            t,
            inner_lop,
            scale,
            gamma
        }
    }
}
impl <'a>  fmt::Debug for ShiftedLinOp <'a>  {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?}, \n", self.t)
    }
}

impl <'a>  LinOp<f64> for ShiftedLinOp<'a>   {
    fn apply_req(
            &self,
            rhs_ncols: usize,
            parallelism: Parallelism,
        ) -> Result<faer::dyn_stack::StackReq, faer::dyn_stack::SizeOverflow> {
        let _ = parallelism;
        let _ = rhs_ncols;
        Ok(faer::dyn_stack::StackReq::empty())
    }

    /// Number of rows in the linop
    fn nrows(&self) -> usize {
        self.inner_lop.nrows()
    }

    /// Number of cols in the linop
    fn ncols(&self) -> usize {
        self.inner_lop.ncols()
    }

    /// Apply linear operator to vec or mat. Stores result in `out`.
    /// Computes (gamma*I + s*J)*v
    /// Where gamma is a shift constant and s is a scaling constant.
    /// By default, s is 1 and gamma is 0.
    /// Ex: implicit methods typically result in s<0, gamma==1.
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Parallelism,
        stack: &mut PodStack,
        )
    {
        // compute unshifted jacobian vector product
        self.inner_lop.apply(out.as_mut(), rhs, parallelism, stack);
        out *= self.scale;

        // compute optional shift
        match self.gamma {
            Some(gamma) => { out += faer::scale(gamma) * rhs.as_ref() },
            (_) => { },
        }
    }

    /// Apply transpose of the linear operator to vec or mat. Stores result in `out`.
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    fn conj_apply(
            &self,
            out: MatMut<'_, f64>,
            rhs: MatRef<'_, f64>,
            parallelism: Parallelism,
            stack: &mut PodStack,
        ) {
        // Not implented error!
        panic!("Not Implemented");
    }
}


/// Linear operator to apply to vec or mat. Stores result in `out`.
/// Provides the linpo L := (gamma*I + scale*J)
/// that be applied to a vector:  L*v
pub struct FdJacLinOp <'a> {
    t: f64,
    x: Mat<f64>,
    frhs: &'a dyn OdeSys<'a>,
    frhs_x: Mat<f64>,
    scale: f64,
    gamma: Option<f64>,
}

impl <'a> FdJacLinOp <'a> {
    /// Create a new finite difference based jacobian linear operator
    ///
    /// * `t` - time at which to evaluate the jacobian
    /// * `x` - current system state about which to evaluate the jacobian
    /// * `frhs` - system rhs
    /// * `scale` - jacobian scale factor
    /// * `gamma` - jacobian shift factor
    pub fn new(t: f64, x: Mat<f64>, frhs: &'a dyn OdeSys<'a> , scale: f64, gamma: Option<f64>)
    -> Self {
        let frhs_x = frhs.frhs(t, x.as_ref());
        Self {
            t,
            x,
            frhs,
            frhs_x,
            scale,
            gamma
        }
    }

    /// Reset point about which to linearize
    ///
    /// * `t` - time at which to evaluate the jacobian
    /// * `x` - current system state about which to evaluate the jacobian
    pub fn set_op_x(&mut self, t: f64, x: Mat<f64>)
    {
        self.t = t;
        self.x = x;
        self.frhs_x = self.frhs.frhs(t, self.x.as_ref());
    }

    /// Reset jacobian scale and diagonal shift
    pub fn set_scale(&mut self, scale: f64, gamma: Option<f64>)
    {
        self.scale = scale;
        self.gamma = gamma;
    }
}

impl <'a> fmt::Debug for FdJacLinOp <'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?}, \n State x={:?} \n f_rhs(x)={:?} \n", self.t, self.x, self.frhs_x)
    }
}

impl <'a> LinOp<f64> for FdJacLinOp <'a> {
    fn apply_req(
            &self,
            rhs_ncols: usize,
            parallelism: Parallelism,
        ) -> Result<faer::dyn_stack::StackReq, faer::dyn_stack::SizeOverflow> {
        let _ = parallelism;
        let _ = rhs_ncols;
        Ok(faer::dyn_stack::StackReq::empty())
    }

    /// Number of rows in the linop
    fn nrows(&self) -> usize {
        self.x.nrows()
    }

    /// Number of cols in the linop
    fn ncols(&self) -> usize {
        self.x.nrows()
    }

    /// Apply linear operator to vec or mat. Stores result in `out`.
    /// Computes (gamma*I + s*J)*v
    /// Where gamma is a shift constant and s is a scaling constant.
    /// By default, s is 1 and gamma is 0.
    /// Ex: implicit methods typically result in s<0, gamma==1.
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Parallelism,
        stack: &mut PodStack,
        )
    {
        // unused
        _ = parallelism;
        _ = stack;

        let x_norm_l1 = self.x.norm_l1();
        let eps = 0.5e-8 * x_norm_l1;
        let ieps = self.scale * 1.0 / eps;
        let x_pert = self.x.as_ref() + faer::scale(eps) * rhs.as_ref();

        // compute unshifted jacobian vector product
        let mut j_v =  (self.frhs.frhs(self.t, x_pert.as_ref()) - self.frhs_x.as_ref()) * faer::scale(ieps) ;

        // compute optional shift
        match self.gamma {
            Some(gamma) => { j_v += faer::scale(gamma) * rhs.as_ref() },
            (_) => { },
        }

        // (gamma*I + scale*J) * v
        out.copy_from(j_v);
    }

    /// Apply transpose of the linear operator to vec or mat. Stores result in `out`.
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    fn conj_apply(
            &self,
            out: MatMut<'_, f64>,
            rhs: MatRef<'_, f64>,
            parallelism: Parallelism,
            stack: &mut PodStack,
        ) {
        // Not implented error!
        panic!("Not Implemented");
    }
}


pub trait OdeSys<'a>: Sync + Send {
    /// Defines the rhs of the system
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64>;
    fn frhs_aug(&self, t: f64, x: MatRef<f64>, aug: MatRef<f64>, aug_scale: f64) -> Mat<f64> {
        aug_scale * self.frhs(t, x) + aug
    }

    /// Defines the Jacobian of the system
    ///
    /// This behavior can be overridden by implementing your own
    /// fjac.
    ///
    /// See: https://stackoverflow.com/questions/59646632/share-function-reference-between-threads-in-rust
    /// for the Sync trait for &dyn Fn for thread safe closures.  This extra Sync trait req
    /// only applys to closures not function pointers!
    fn fjac<'b>(&'a self,
            t: f64,
            x: MatRef<'b, f64>)
        -> Box<dyn LinOp<f64> + 'a>;

    fn fjac_shifted<'b>(&'a self,
            t: f64,
            x: MatRef<'b, f64>,
            scale: f64,
            gamma: Option<f64>)
        -> ShiftedLinOp<'a>
    {
        ShiftedLinOp::new(
            t,
            self.fjac(t, x),
            scale,
            gamma,
        )
    }
}

/// Obtain finite difference jacobian LinOp of a system at a given operating point
pub fn get_fd_jac<'a>(sys: &'a dyn OdeSys<'a>, t: f64, x: MatRef<f64>) -> FdJacLinOp<'a>
{
    // sys.fjac(t, x)
    FdJacLinOp::new(t, x.to_owned(), sys, 1.0, None)
}

/// Obtain finite difference shifted and scaled jacobian LinOp of a system at a given operating point
pub fn get_fd_jac_shifted<'a>(inner_lop: Box<dyn LinOp<f64> + 'a>, t: f64, scale: f64, gamma: Option<f64>) -> ShiftedLinOp<'a>
{
    // sys.fjac_shifted(t, x, scale, gamma)
    ShiftedLinOp::new(t, inner_lop, scale, gamma)
}
