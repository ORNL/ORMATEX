/// Defines and ODE system of equations
/// Defines interface for integration ode equations with
/// exponential integrators, implicit and explicit integrators
///
use faer::prelude::*;
use faer::matrix_free::LinOp;
use faer::Par;
use faer::dyn_stack::PodStack;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer_traits::math_utils::abs;
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


#[derive(Clone)]
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
    fn step(&self, dt: Self::TimeType) -> Result<StepResult<Self::TimeType, Self::SysStateType>, StepError>;

    /// Get current time
    fn time(&self) -> Self::TimeType;

    /// Get current system state
    fn state(&self) -> Self::SysStateType;

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
    lop.apply(
        out.as_mut(),
        q,
        faer::get_global_parallelism(),
        MemStack::new(&mut MemBuffer::new(StackReq::empty()))
        );
    out
}


/// Wrapper to extend a LinOp, A
/// and applies
/// [[ A,  B],
///  [ 0,  K]]
/// to a vector.
///
/// example use:
/// let elop = ExtendedLinOp::new(lop, &vb);
/// let v, n = elop.get_v(&vb);
/// let mut res = faer::Mat::zeros(n, 1);
/// elop.apply(res.as_mut(), v.as_ref(), ..);
pub struct ExtendedLinOp<'a> {
    t: f64,
    inner_lop: Box<dyn LinOp<f64> + 'a>,
    bmat: faer::Mat<f64>,
    kmat: faer::Mat<f64>,
}

impl <'a> ExtendedLinOp<'a> {
    pub fn new(t: f64, inner_lop: Box<dyn LinOp<f64> + 'a>, vb: &Vec<MatRef<f64>>) -> Self {
        let n = vb[0].nrows();
        let p = vb.len() - 1;
        let mut bmat = faer::Mat::zeros(n, p);
        let mut i = 0;
        // reverse iter through vb
        for k in (1..p).rev() {
            bmat.as_mut().get_mut(.., i..i+1).copy_from(
                vb[k].as_ref());
            i += 1;
        }
        let mut kmat = faer::Mat::zeros(p, p);
        kmat.as_mut().get_mut(0..p-1, 1..).copy_from(
            faer::Mat::<f64>::identity(p-1, p-1));
        Self {
            t,
            inner_lop,
            bmat,
            kmat
        }
    }

    /// helper method to create rhs vector for this extended linop
    pub fn get_v(vb: &Vec<MatRef<f64>>) -> (Mat<f64>, usize) {
        let n = vb[0].nrows();
        let p = vb.len() - 1;
        // let mut unit_vec = faer::Mat::zeros(p, 1);
        // unit_vec[(n, 0)] = 1.0;
        let mut out: Mat<f64> = faer::Mat::zeros(n+p, 1);
        out[(n+p, 0)] = 1.0;
        out.as_mut().get_mut(0..n, 0..1).copy_from(vb[0].as_ref());
        (out, n)
    }
}

impl <'a>  fmt::Debug for ExtendedLinOp <'a>  {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?}, \n", self.t)
    }
}

impl <'a> LinOp<f64> for ExtendedLinOp<'a>   {
    fn apply_scratch(
            &self,
            rhs_ncols: usize,
            parallelism: Par,
        ) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        StackReq::empty()
    }

    /// Number of rows in the linop
    fn nrows(&self) -> usize {
        self.inner_lop.nrows()
    }

    /// Number of cols in the linop
    fn ncols(&self) -> usize {
        self.inner_lop.ncols()
    }

    /// Apply the extended lop
    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        let n = self.bmat.nrows();
        let p = self.bmat.ncols();

        let mut av = faer::Mat::zeros(rhs.nrows(), rhs.ncols());
        self.inner_lop.apply(
            av.as_mut(),
            rhs.get(0..n, ..),
            parallelism,
            stack);
        let ab_v = faer::Scale(self.t) * av +
            self.bmat.as_ref() * rhs.get(p..rhs.nrows(), ..);
        let k_v = self.kmat.as_ref() * rhs.get(p..rhs.nrows(), ..);
        out.as_mut().get_mut(0..ab_v.nrows(), ..).copy_from(ab_v.as_ref());
        out.as_mut().get_mut(ab_v.nrows().., ..).copy_from(k_v);
    }

    fn conj_apply(
            &self,
            out: MatMut<'_, f64>,
            rhs: MatRef<'_, f64>,
            parallelism: Par,
            stack: &mut MemStack,
        ) {
        // Not implented error!
        panic!("Not Implemented");
    }
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
    fn apply_scratch(
            &self,
            rhs_ncols: usize,
            parallelism: Par,
        ) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        StackReq::empty()
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
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        // compute unshifted jacobian vector product
        self.inner_lop.apply(out.as_mut(), rhs, parallelism, stack);
        out *= self.scale;

        // compute optional shift
        match self.gamma {
            Some(gamma) => { out += faer::Scale(gamma) * rhs.as_ref() },
            _ => { },
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
            parallelism: Par,
            stack: &mut MemStack,
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
    fn apply_scratch(
            &self,
            rhs_ncols: usize,
            parallelism: Par,
        ) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        StackReq::empty()
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
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        // unused
        _ = parallelism;
        _ = stack;

        let x_norm_l1 = self.x.norm_max().abs();
        let eps = 0.5e-8 * x_norm_l1;
        let ieps = self.scale * 1.0 / eps;
        let x_pert = self.x.as_ref() + faer::Scale(eps) * rhs.as_ref();

        // compute unshifted jacobian vector product
        let mut j_v = (self.frhs.frhs(self.t, x_pert.as_ref())-self.frhs_x.as_ref())*faer::Scale(ieps);

        // compute optional shift
        match self.gamma {
            Some(gamma) => { j_v += faer::Scale(gamma) * rhs.as_ref() },
            _ => { },
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
            parallelism: Par,
            stack: &mut MemStack,
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
