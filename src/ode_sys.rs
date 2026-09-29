/*
 * Copyright(c) 2025 UT-Battelle, LLC
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
//! Definition of the ODE system interface and supporting linear operators.
//!
//! This module defines the [`OdeSys`] trait, which users implement to describe
//! an initial value problem
//!
//! $$ M \thinspace \frac{dy}{dt} = f(t, y), \qquad y(t_0) = y_0 $$
//!
//! (with $M = I$ unless a mass matrix is supplied) so that it can be advanced by
//! the exponential, implicit and explicit integrators in this crate. Note that
//! a non-identity mass matrix is only supported by the implicit integrators,
//! see [`OdeSys::fmass`].
//!
//! It also provides the shared building blocks used by the integrators:
//!
//! * [`StepResult`] and [`StepError`] - the result and error of a single time step.
//! * [`FdJacLinOp`] - a matrix-free finite difference Jacobian-vector product.
//! * [`ShiftedLinOp`] - the shifted and scaled operator $\gamma M + s J$ used by
//!   implicit methods.
//! * [`ExtendedLinOp`] and [`DynRefExtendedLinOp`] - augmented operators that allow
//!   a linear combination of $\varphi_k$ function products to be evaluated with a
//!   single matrix exponential action.
//! * [`get_fd_jac`], [`get_fd_jac_shifted`] and [`apply_linop`] - helper functions.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::Par;
use std::{error::Error, fmt};

/// Error returned when a time integrator fails to produce a step.
///
/// Its `Display` implementation prints only the fixed text `StepError`;
/// use the `error_code` and `msg` fields for details.
#[derive(Debug)]
pub struct StepError {
    /// Integrator specific error code
    pub error_code: usize,
    /// Human readable description of the failure
    pub msg: String,
}

impl Error for StepError {}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StepError")
    }
}

/// Result of a single (proposed) time step.
///
/// Returned by [`crate::ode_traits::IntegrateSys::step`]. The step is only
/// recorded by the integrator once it is passed to `accept_step`.
///
/// # Type parameters
///
/// * `T` - time type (typically `f64`)
/// * `S` - system state type (typically `faer::Mat<f64>`)
#[derive(Clone)]
pub struct StepResult<T, S> {
    /// System time at the end of the step, $t_{n+1}$
    pub t: T,
    /// Time step size taken, $\Delta t$
    pub dt: T,
    /// System state at the end of the step, $y_{n+1}$
    pub y: S,
    /// Embedded error estimate. `Some` only if the method provides an embedded
    /// error estimate, otherwise `None`.
    pub err: Option<f64>,
}
impl<T, S> StepResult<T, S> {
    /// Create a new step result.
    ///
    /// # Arguments
    ///
    /// * `t` - system time at the end of the step
    /// * `dt` - time step size taken
    /// * `y` - system state at the end of the step
    /// * `err` - embedded error estimate, if the method provides one
    pub fn new(t: T, dt: T, y: S, err: Option<f64>) -> Self {
        Self { t, dt, y, err }
    }
}

/// Apply a linear operator to a matrix or vector and return the result.
///
/// Convenience helper that allocates the output matrix (and uses faer's global
/// parallelism setting), so it performs an extra allocation compared to calling
/// `LinOp::apply` directly.
///
/// # Arguments
///
/// * `lop` - linear operator $A$
/// * `q` - matrix or column vector to apply the operator to
///
/// # Returns
///
/// The product $A q$ as a new matrix with `lop.nrows()` rows and `q.ncols()` columns.
pub fn apply_linop(lop: &impl LinOp<f64>, q: MatRef<f64>) -> Mat<f64> {
    let mut out = faer::Mat::zeros(lop.nrows(), q.ncols());
    lop.apply(
        out.as_mut(),
        q,
        faer::get_global_parallelism(),
        MemStack::new(&mut MemBuffer::new(StackReq::empty())),
    );
    out
}

/// Augmented linear operator that owns its inner operator.
///
/// Wraps a linear operator $A$ ($n \times n$) and applies the augmented
/// $(n+p) \times (n+p)$ operator
///
/// ```text
/// [ t*A   B ]
/// [  0    K ]
/// ```
///
/// to a vector. Here `t` is a scale factor for the inner operator (typically
/// the step size), $B$ is an $n \times p$ matrix whose columns are built from
/// `vb[1..]` (in reverse order, so column `p-1` is `vb[1]` and column `0` is
/// `vb[p]`), and $K$ is the $p \times p$ shift matrix with ones on the first
/// superdiagonal. Applying a matrix exponential to the augmented operator lets a
/// linear combination of $\varphi_k$ products
/// $\sum_k \varphi_k(t A) v_k$ be evaluated with one exponential action.
///
/// Use [`ExtendedLinOp::get_v`] to build the matching starting vector.
///
/// Example use:
///
/// ```ignore
/// let elop = ExtendedLinOp::new(dt, lop, &vb);
/// let (v, n) = elop.get_v(&vb);
/// let mut res = faer::Mat::zeros(n, 1);
/// elop.apply(res.as_mut(), v.as_ref(), ..);
/// ```
///
/// Note: `nrows`/`ncols` of this operator report the size of the inner
/// operator only (not $n+p$); see [`DynRefExtendedLinOp`] for a variant that
/// reports the extended size.
pub struct ExtendedLinOp<'a> {
    t: f64,
    inner_lop: Box<dyn LinOp<f64> + 'a>,
    bmat: faer::Mat<f64>,
    kmat: faer::Mat<f64>,
}

impl<'a> ExtendedLinOp<'a> {
    /// Build the extended operator.
    ///
    /// # Arguments
    ///
    /// * `t` - scale factor applied to the inner operator (typically the step size)
    /// * `inner_lop` - the inner $n \times n$ linear operator $A$
    /// * `vb` - vectors `[v0, v1, ..., vp]`, each of length $n$. `v0` is only used for
    ///   its number of rows here; `v1..vp` fill the columns of $B$.
    ///
    /// # Panics
    ///
    /// Panics if `vb` is empty or contains only one vector ($p = 0$).
    pub fn new(t: f64, inner_lop: Box<dyn LinOp<f64> + 'a>, vb: &Vec<MatRef<f64>>) -> Self {
        let n = vb[0].nrows();
        let p = vb.len() - 1;
        let mut bmat = faer::Mat::zeros(n, p);
        let mut i = 1;
        // build extended linear operator blocks
        for k in (0..p).rev() {
            bmat.as_mut()
                .get_mut(.., k..k + 1)
                .copy_from(vb[i].as_ref());
            i += 1;
        }
        let mut kmat = faer::Mat::zeros(p, p);
        kmat.as_mut()
            .get_mut(0..p - 1, 1..)
            .copy_from(faer::Mat::<f64>::identity(p - 1, p - 1));
        Self {
            t,
            inner_lop,
            bmat,
            kmat,
        }
    }

    /// Create the starting vector for this extended linop.
    ///
    /// The result has length $n+p$: the first $n$ entries are `vb[0]`, the
    /// following $p-1$ entries are zero and the last entry is one.
    ///
    /// # Arguments
    ///
    /// * `vb` - vectors `[v0, v1, ..., vp]` used to build the operator
    ///
    /// # Returns
    ///
    /// A tuple `(v, n)` of the extended vector and the size $n$ of the original system.
    pub fn get_v(&self, vb: &Vec<MatRef<f64>>) -> (Mat<f64>, usize) {
        let n = vb[0].nrows();
        let p = vb.len() - 1;
        // let mut unit_vec = faer::Mat::zeros(p, 1);
        // unit_vec[(n, 0)] = 1.0;
        let mut out: Mat<f64> = faer::Mat::zeros(n + p, 1);
        out[(n + p - 1, 0)] = 1.0;
        out.as_mut().get_mut(0..n, 0..1).copy_from(vb[0].as_ref());
        (out, n)
    }
}

impl<'a> fmt::Debug for ExtendedLinOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?}, \n", self.t)
    }
}

impl<'a> LinOp<f64> for ExtendedLinOp<'a> {
    fn apply_scratch(&self, rhs_ncols: usize, parallelism: Par) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        self.inner_lop.apply_scratch(rhs_ncols, parallelism)
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
    ) {
        let n = self.bmat.nrows();
        let p = self.bmat.ncols();

        let mut av = faer::Mat::zeros(n, rhs.ncols());
        self.inner_lop
            .apply(av.as_mut(), rhs.get(0..n, ..), parallelism, stack);
        let ab_v = faer::Scale(self.t) * av + self.bmat.as_ref() * rhs.get(rhs.nrows() - p.., ..);
        let k_v = self.kmat.as_ref() * rhs.get(rhs.nrows() - p.., ..);
        out.as_mut()
            .get_mut(0..ab_v.nrows(), ..)
            .copy_from(ab_v.as_ref());
        out.as_mut().get_mut(ab_v.nrows().., ..).copy_from(k_v);
    }

    fn conj_apply(
        &self,
        _out: MatMut<'_, f64>,
        _rhs: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        // Not implemented error!
        panic!("Not Implemented");
    }
}

/// Augmented linear operator that borrows its inner operator.
///
/// Same construction as [`ExtendedLinOp`] (the augmented operator
/// with blocks `t*A`, $B$, $0$, $K$), but the inner operator is held by
/// reference as a `&dyn LinOp<f64>`, and `nrows`/`ncols` report the extended
/// size $n+p$. This is the operator passed to the phi-function evaluators
/// in [`crate::matexp_traits::LinOpPhikvEvaluator`].
pub struct DynRefExtendedLinOp<'a> {
    t: f64,
    inner_lop: &'a dyn LinOp<f64>,
    bmat: faer::Mat<f64>,
    kmat: faer::Mat<f64>,
}

impl<'a> DynRefExtendedLinOp<'a> {
    /// Build the extended operator.
    ///
    /// # Arguments
    ///
    /// * `t` - scale factor applied to the inner operator (typically the step size)
    /// * `inner_lop` - the inner $n \times n$ linear operator $A$, borrowed
    /// * `vb` - vectors `[v0, v1, ..., vp]`, each of length $n$. `v0` is only used for
    ///   its number of rows here; `v1..vp` fill the columns of the block $B$.
    ///
    /// # Panics
    ///
    /// Panics if `vb` is empty.
    pub fn new(t: f64, inner_lop: &'a dyn LinOp<f64>, vb: &Vec<MatRef<f64>>) -> Self {
        let n = vb[0].nrows();
        let p = vb.len() - 1;
        let mut bmat = faer::Mat::zeros(n, p);
        let mut i = 1;
        // build extended linear operator blocks
        for k in (0..p).rev() {
            bmat.as_mut()
                .get_mut(.., k..k + 1)
                .copy_from(vb[i].as_ref());
            i += 1;
        }
        let mut kmat = faer::Mat::zeros(p, p);
        if p > 0 {
            kmat.as_mut()
                .get_mut(0..p - 1, 1..)
                .copy_from(faer::Mat::<f64>::identity(p - 1, p - 1));
        }
        Self {
            t,
            inner_lop,
            bmat,
            kmat,
        }
    }

    /// Create the starting vector for this extended linop.
    ///
    /// The result has length $n+p$: the first $n$ entries are `vb[0]`, the
    /// following $p-1$ entries are zero and the last entry is one.
    ///
    /// # Arguments
    ///
    /// * `vb` - vectors `[v0, v1, ..., vp]` used to build the operator
    ///
    /// # Returns
    ///
    /// A tuple `(v, n)` of the extended vector and the size $n$ of the original system.
    pub fn get_v(&self, vb: &Vec<MatRef<f64>>) -> (Mat<f64>, usize) {
        let n = vb[0].nrows();
        let p = vb.len() - 1;
        // let mut unit_vec = faer::Mat::zeros(p, 1);
        // unit_vec[(n, 0)] = 1.0;
        let mut out: Mat<f64> = faer::Mat::zeros(n + p, 1);
        out[(n + p - 1, 0)] = 1.0;
        out.as_mut().get_mut(0..n, 0..1).copy_from(vb[0].as_ref());
        (out, n)
    }
}

impl<'a> fmt::Debug for DynRefExtendedLinOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?}, \n", self.t)
    }
}

impl<'a> LinOp<f64> for DynRefExtendedLinOp<'a> {
    fn apply_scratch(&self, rhs_ncols: usize, parallelism: Par) -> StackReq {
        // let _ = parallelism;
        // let _ = rhs_ncols;
        // StackReq::empty()
        self.inner_lop.apply_scratch(rhs_ncols, parallelism)
    }

    /// Number of rows in the linop
    fn nrows(&self) -> usize {
        self.inner_lop.nrows() + self.kmat.nrows()
    }

    /// Number of cols in the linop
    fn ncols(&self) -> usize {
        self.inner_lop.ncols() + self.kmat.ncols()
    }

    /// Apply the extended lop
    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Par,
        stack: &mut MemStack,
    ) {
        let n = self.bmat.nrows();
        let p = self.bmat.ncols();

        let mut av = faer::Mat::zeros(n, rhs.ncols());
        self.inner_lop
            .apply(av.as_mut(), rhs.get(0..n, ..), parallelism, stack);
        let ab_v = faer::Scale(self.t) * av + self.bmat.as_ref() * rhs.get(rhs.nrows() - p.., ..);
        let k_v = self.kmat.as_ref() * rhs.get(rhs.nrows() - p.., ..);
        out.as_mut()
            .get_mut(0..ab_v.nrows(), ..)
            .copy_from(ab_v.as_ref());
        out.as_mut().get_mut(ab_v.nrows().., ..).copy_from(k_v);
    }

    fn conj_apply(
        &self,
        _out: MatMut<'_, f64>,
        _rhs: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        // Not implemented error!
        panic!("Not Implemented");
    }
}

/// Wrapper to shift and scale a linear operator, optionally weighted by a mass matrix.
///
/// Computes one of:
///
/// * $(\gamma M + s J) v$ when `gamma` is `Some` and `mass` is `Some`
/// * $(\gamma I + s J) v$ when `gamma` is `Some` and `mass` is `None` (default)
/// * $(s J) v$ when `gamma` is `None`
///
/// where `J` is the wrapped inner linear operator (the system Jacobian),
/// `M` is an optional mass matrix supplied by `OdeSys::fmass`, `s` is the
/// `scale` factor, and `gamma` is the `gamma` shift value.
///
/// Implicit integrators solve `(gamma*M + s*J)*delta = r`, so setting `scale = -dt*a_ii`
/// and `gamma = 1` yields the standard `(M - dt*a_ii*J)` system.
///
/// The mass matrix only weights the shift term. It does not multiply $J$, and
/// it is ignored when `gamma` is `None`. This operator is the exact Newton
/// matrix for a residual $G(y) = M (y - y_{expl}) - \Delta t \thinspace a_{ii} f(t, y)$,
/// which is what the implicit integrators in this crate use, see
/// [`OdeSys::fmass`] and [`crate::ode_implicit`].
pub struct ShiftedLinOp<'a> {
    t: f64,
    inner_lop: Box<dyn LinOp<f64> + 'a>,
    scale: f64,
    gamma: Option<f64>,
    /// Optional mass matrix M.  `None` falls back to the identity (current
    /// behaviour unchanged).  When `Some`, the gamma shift uses `gamma*M*v`
    /// instead of `gamma*I*v`.
    mass: Option<Box<dyn LinOp<f64> + 'a>>,
}

impl<'a> ShiftedLinOp<'a> {
    /// Create a shifted and scaled operator.
    ///
    /// # Arguments
    ///
    /// * `t` - time at which the operator is evaluated (informational only)
    /// * `inner_lop` - the wrapped operator $J$, typically the system Jacobian
    /// * `scale` - scale factor $s$ applied to $J$
    /// * `gamma` - shift factor $\gamma$; `None` disables the shift entirely
    /// * `mass` - optional mass matrix $M$; `None` uses the identity
    pub fn new(
        t: f64,
        inner_lop: Box<dyn LinOp<f64> + 'a>,
        scale: f64,
        gamma: Option<f64>,
        mass: Option<Box<dyn LinOp<f64> + 'a>>,
    ) -> Self {
        Self {
            t,
            inner_lop,
            scale,
            gamma,
            mass,
        }
    }
}
impl<'a> fmt::Debug for ShiftedLinOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?}, \n", self.t)
    }
}

impl<'a> LinOp<f64> for ShiftedLinOp<'a> {
    fn apply_scratch(&self, rhs_ncols: usize, parallelism: Par) -> StackReq {
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
    ///
    /// Computes `(gamma*M + s*J)*v`, reducing to `(gamma*I + s*J)*v` when `mass`
    /// is `None` (unchanged from the previous identity-shift behaviour).
    ///
    /// # Arguments
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    /// * `stack` - faer scratch memory
    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Par,
        stack: &mut MemStack,
    ) {
        // s*J*v
        self.inner_lop.apply(out.as_mut(), rhs, parallelism, stack);
        out *= self.scale;

        // gamma*M*v  or  gamma*v  (identity fallback)
        match self.gamma {
            Some(gamma) => {
                match &self.mass {
                    Some(mass_lop) => {
                        // Allocate a temporary for M*v and accumulate gamma*M*v.
                        let mut mv = faer::Mat::zeros(out.nrows(), rhs.ncols());
                        mass_lop.apply(mv.as_mut(), rhs, parallelism, stack);
                        out += faer::Scale(gamma) * mv.as_ref();
                    }
                    None => {
                        // No mass matrix: gamma*I*v = gamma*v  (original behaviour).
                        out += faer::Scale(gamma) * rhs.as_ref();
                    }
                }
            }
            _ => {}
        }
    }

    /// Apply transpose of the linear operator to vec or mat. Stores result in `out`.
    ///
    /// Not implemented, always panics.
    ///
    /// # Arguments
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    fn conj_apply(
        &self,
        _out: MatMut<'_, f64>,
        _rhs: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        // Not implemented error!
        panic!("Not Implemented");
    }
}

/// Matrix-free finite difference Jacobian linear operator.
///
/// Provides the operator $L = \gamma I + s J$ that can be applied to a vector
/// as $L v$, where $J = \partial f / \partial x$ evaluated at the stored point
/// $(t, x)$ is never formed. Each product is approximated by a forward difference
///
/// $$ J v \approx \frac{f(t, x + \epsilon v) - f(t, x)}{\epsilon},
/// \qquad \epsilon = 5 \times 10^{-9} \thinspace \max_i |x_i| $$
///
/// which costs one extra rhs evaluation per column of `v`. Note that $\epsilon$
/// is not scaled by the norm of `v` and vanishes if $x = 0$. The mass matrix
/// is not used by this operator.
pub struct FdJacLinOp<'a> {
    t: f64,
    x: Mat<f64>,
    frhs: &'a dyn OdeSys<'a>,
    frhs_x: Mat<f64>,
    scale: f64,
    gamma: Option<f64>,
}

impl<'a> FdJacLinOp<'a> {
    /// Create a new finite difference based jacobian linear operator.
    ///
    /// Evaluates the system rhs once at $(t, x)$ and caches it.
    ///
    /// # Arguments
    ///
    /// * `t` - time at which to evaluate the jacobian
    /// * `x` - current system state about which to evaluate the jacobian
    /// * `frhs` - system rhs
    /// * `scale` - jacobian scale factor $s$
    /// * `gamma` - jacobian shift factor $\gamma$; `None` for no shift
    pub fn new(
        t: f64,
        x: Mat<f64>,
        frhs: &'a dyn OdeSys<'a>,
        scale: f64,
        gamma: Option<f64>,
    ) -> Self {
        let frhs_x = frhs.frhs(t, x.as_ref());
        Self {
            t,
            x,
            frhs,
            frhs_x,
            scale,
            gamma,
        }
    }

    /// Reset point about which to linearize.
    ///
    /// Re-evaluates and caches the system rhs at the new point.
    ///
    /// # Arguments
    ///
    /// * `t` - time at which to evaluate the jacobian
    /// * `x` - current system state about which to evaluate the jacobian
    pub fn set_op_x(&mut self, t: f64, x: Mat<f64>) {
        self.t = t;
        self.x = x;
        self.frhs_x = self.frhs.frhs(t, self.x.as_ref());
    }

    /// Reset jacobian scale and diagonal shift.
    ///
    /// # Arguments
    ///
    /// * `scale` - jacobian scale factor $s$
    /// * `gamma` - jacobian shift factor $\gamma$; `None` for no shift
    pub fn set_scale(&mut self, scale: f64, gamma: Option<f64>) {
        self.scale = scale;
        self.gamma = gamma;
    }
}

impl<'a> fmt::Debug for FdJacLinOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "t={:?}, \n State x={:?} \n f_rhs(x)={:?} \n",
            self.t, self.x, self.frhs_x
        )
    }
}

impl<'a> LinOp<f64> for FdJacLinOp<'a> {
    fn apply_scratch(&self, rhs_ncols: usize, parallelism: Par) -> StackReq {
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
    ///
    /// Computes $(\gamma I + s J) v$ where $\gamma$ is a shift constant and $s$
    /// is a scaling constant. By default, $s$ is 1 and there is no shift.
    /// For example, implicit methods typically use $s < 0$ and $\gamma = 1$.
    ///
    /// # Arguments
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism (unused)
    /// * `stack` - faer scratch memory (unused)
    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Par,
        stack: &mut MemStack,
    ) {
        // unused
        _ = parallelism;
        _ = stack;

        let x_norm_l1 = self.x.norm_max().abs();
        let eps = 0.5e-8 * x_norm_l1;
        let ieps = self.scale * 1.0 / eps;

        for j in 0..out.ncols() {
            let x_pert = self.x.as_ref() + faer::Scale(eps) * rhs.col(j).as_mat();

            // compute unshifted jacobian vector product
            let mut j_v = (self.frhs.frhs(self.t, x_pert.as_ref()) - self.frhs_x.as_ref())
                * faer::Scale(ieps);

            // compute optional shift
            match self.gamma {
                Some(gamma) => j_v += faer::Scale(gamma) * rhs.col(j).as_mat(),
                _ => {}
            }

            // (gamma*I + scale*J) * v
            out.as_mut().col_mut(j).copy_from(j_v.col(0));
        }
    }

    /// Apply transpose of the linear operator to vec or mat. Stores result in `out`.
    ///
    /// Not implemented, always panics.
    ///
    /// # Arguments
    ///
    /// * `out` - output
    /// * `rhs` - target to apply linop to
    /// * `parallelism` - faer parallelism
    fn conj_apply(
        &self,
        _out: MatMut<'_, f64>,
        _rhs: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
    ) {
        // Not implemented error!
        panic!("Not Implemented");
    }
}

/// Interface describing an ODE system, implemented by the user.
///
/// A system represents the initial value problem
///
/// $$ M(t) \thinspace \frac{dy}{dt} = f(t, y), \qquad y(t_0) = y_0 $$
///
/// where $y \in \mathbb{R}^n$ is the state, $f$ is the right hand side
/// (see [`OdeSys::frhs`]) and $M$ is an optional mass matrix (see
/// [`OdeSys::fmass`]; the identity by default).
///
/// Methods required from the implementor:
///
/// * [`OdeSys::frhs`] evaluates $f(t, y)$.
/// * [`OdeSys::fjac`] returns the Jacobian $J = \partial f / \partial y$ at
///   $(t, y)$ as a matrix-free linear operator. Use [`get_fd_jac`] for a finite
///   difference approximation if an analytic Jacobian is not available.
///
/// Optional methods with default implementations:
///
/// * [`OdeSys::fmass`] returns the mass matrix $M$. Default: none (identity).
/// * [`OdeSys::fjac_shifted`] returns the operator $\gamma M + s J$ used by
///   implicit integrators. Default: built from `fjac` and `fmass`.
///
/// The implicit integrators support a mass matrix; the explicit and exponential
/// integrators ignore it and integrate $y^\prime = f(t, y)$. See [`OdeSys::fmass`].
///
/// The trait requires `Sync + Send`. The lifetime parameter `'a` is the lifetime
/// of the borrow of the system by the returned linear operators.
///
/// State vectors are `faer` matrices with $n$ rows. Integrators use a single
/// column.
///
/// # Example
///
/// A linear decay system $dy/dt = -k y$ using a finite difference Jacobian:
///
/// ```ignore
/// use faer::prelude::*;
/// use faer::matrix_free::LinOp;
/// use ormatex::ode_sys::{get_fd_jac, OdeSys};
///
/// struct Decay {
///     k: f64,
/// }
///
/// impl<'a> OdeSys<'a> for Decay {
///     fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
///         Scale(-self.k) * x
///     }
///
///     fn fjac<'b>(&'a self, t: f64, x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
///         Box::new(get_fd_jac(self, t, x))
///     }
/// }
/// ```
pub trait OdeSys<'a>: Sync + Send {
    /// Evaluate the right hand side $f(t, y)$ of the system.
    ///
    /// # Arguments
    ///
    /// * `t` - the current time
    /// * `x` - the current state $y$ ($n$ rows)
    ///
    /// # Returns
    ///
    /// The value $f(t, y)$, a matrix of the same shape as `x`.
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64>;

    /// Return the Jacobian $J = \partial f / \partial y$ of the system at $(t, y)$
    /// as a matrix-free linear operator.
    ///
    /// Implement this with an analytic Jacobian where possible. Otherwise
    /// [`get_fd_jac`] provides a finite difference approximation based on
    /// [`OdeSys::frhs`].
    ///
    /// # Arguments
    ///
    /// * `t` - the current time
    /// * `x` - the current state
    ///
    /// # Returns
    ///
    /// A boxed linear operator $J$ that applies the Jacobian to vectors.
    fn fjac<'b>(&'a self, t: f64, x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a>;

    /// Optional mass matrix $M$ at time `t`.
    ///
    /// When `Some(M)` is returned, the shifted Jacobian operator used by
    /// implicit integrators becomes $\gamma M + s J$ instead of $\gamma I + s J$.
    /// This lets the user solve DAE-like or FEM problems where the time
    /// derivative appears as $M \thinspace dy/dt = f(t, y)$.
    ///
    /// The default implementation returns `None`, which preserves
    /// identity-matrix behaviour.
    ///
    /// # Support in the integrators
    ///
    /// * The implicit integrators (DIRK, BDF; see [`crate::ode_implicit`]) solve
    ///   $M y^\prime = f(t, y)$ correctly, including a singular $M$ for methods
    ///   whose stages are all implicit.
    /// * The explicit (Runge-Kutta) and exponential (EPI, EXPRB) integrators
    ///   never call `fmass` and integrate $y^\prime = f(t, y)$. To use them with
    ///   a nonsingular mass matrix, fold $M^{-1}$ into [`OdeSys::frhs`] and
    ///   [`OdeSys::fjac`] and return `None` here.
    ///
    /// # Arguments
    ///
    /// * `t` - the current time
    ///
    /// # Returns
    ///
    /// `Some(M)` as a boxed linear operator, or `None` for the identity.
    fn fmass(&'a self, _t: f64) -> Option<Box<dyn LinOp<f64> + 'a>> {
        None
    }

    /// Return the shifted and scaled operator $W = \gamma M + s J$.
    ///
    /// Implicit integrators use $\gamma = 1$ and $s = -\Delta t \thinspace a_{ii}$ so
    /// that $W = M - \Delta t \thinspace a_{ii} J$ is the Newton matrix of the residual
    /// $M (Y - Y_{expl}) - \Delta t \thinspace a_{ii} f(t, Y)$.
    ///
    /// If no mass matrix is supplied by [`OdeSys::fmass`], this is
    /// $\gamma I + s J$. If `gamma` is `None` the shift is omitted and the
    /// operator is $s J$. The default implementation wraps [`OdeSys::fjac`]
    /// and [`OdeSys::fmass`] in a [`ShiftedLinOp`].
    ///
    /// # Arguments
    ///
    /// * `t` - the current time
    /// * `x` - the current state
    /// * `scale` - scale factor $s$ applied to the Jacobian
    /// * `gamma` - shift factor $\gamma$, or `None` for no shift
    ///
    /// # Returns
    ///
    /// The operator $W$.
    fn fjac_shifted<'b>(
        &'a self,
        t: f64,
        x: MatRef<'b, f64>,
        scale: f64,
        gamma: Option<f64>,
    ) -> ShiftedLinOp<'a> {
        ShiftedLinOp::new(t, self.fjac(t, x), scale, gamma, self.fmass(t))
    }
}

/// Obtain a finite difference jacobian operator of a system at a given operating point.
///
/// Returns an [`FdJacLinOp`] with scale 1 and no shift, i.e. $J v$. Evaluates
/// the system rhs once at $(t, x)$.
///
/// # Arguments
///
/// * `sys` - the ODE system
/// * `t` - time at which to evaluate the jacobian
/// * `x` - state about which to linearize
pub fn get_fd_jac<'a>(sys: &'a dyn OdeSys<'a>, t: f64, x: MatRef<f64>) -> FdJacLinOp<'a> {
    // sys.fjac(t, x)
    FdJacLinOp::new(t, x.to_owned(), sys, 1.0, None)
}

/// Wrap a jacobian operator in a shifted and scaled operator.
///
/// Returns the operator $\gamma I + s J$ (or $s J$ if `gamma` is `None`) with
/// $J$ given by `inner_lop`. No mass matrix is applied, since no `OdeSys` is
/// available here; use [`OdeSys::fjac_shifted`] if a mass matrix is required.
///
/// # Arguments
///
/// * `inner_lop` - the jacobian operator $J$ (for example from [`get_fd_jac`])
/// * `t` - time at which the operator is evaluated
/// * `scale` - scale factor $s$ applied to $J$
/// * `gamma` - shift factor $\gamma$, or `None` for no shift
pub fn get_fd_jac_shifted<'a>(
    inner_lop: Box<dyn LinOp<f64> + 'a>,
    t: f64,
    scale: f64,
    gamma: Option<f64>,
) -> ShiftedLinOp<'a> {
    // No OdeSys available here, so mass matrix is always None.
    // Use OdeSys::fjac_shifted if a mass matrix is required.
    ShiftedLinOp::new(t, inner_lop, scale, gamma, None)
}
