//! # ORMATEX
//!
//! **O**ak **R**idge **MAT**rix **EX**ponential tools.
//!
//! ORMATEX computes the matrix exponential $\exp(A t)$, its action on a vector
//! $\exp(A t) v_0$, and the related $\varphi_k$-functions
//!
//! $$ \varphi_0(z) = e^z, \qquad
//!    \varphi_{k+1}(z) = \frac{\varphi_k(z) - 1/k!}{z}. $$
//!
//! For small dense matrices these are evaluated directly. For large and sparse
//! (or matrix-free) operators $A$, Krylov subspace and Leja polynomial methods
//! evaluate the vector products $\varphi_k(t A) v$ using only matrix-vector
//! products with $A$.
//!
//! On top of these kernels the crate provides exponential time integrators for
//! large systems of coupled ODEs
//!
//! $$ \frac{dy}{dt} = f(t, y), \qquad y(t_0) = y_0, $$
//!
//! together with classic explicit and implicit Runge-Kutta and BDF integrators
//! for comparison. The linear algebra is built on [`faer`].
//!
//! ORMATEX is a mixed Rust and Python package. The Rust integrators are also
//! available from Python through the `ormatex_rspy` module (cargo feature
//! `python`, enabled by default); see the project `README.md` for the Python
//! interface, which additionally offers JAX based integrators.
//!
//! ## Matrix exponential and phi-function evaluators
//!
//! | Module | Method | Best suited for |
//! | ------ | ------ | --------------- |
//! | [`matexp_pade`] | Pade approximation with scaling and squaring (Higham) | small dense matrices, real or complex |
//! | [`matexp_taylor`] | Taylor series | small dense matrices, phi-functions |
//! | [`matexp_cauchy`] | Contour integral (CRAM, parabolic contour) via partial fractions | dense matrices with spectrum near the negative real axis |
//! | [`matexp_krylov`] | Krylov subspace (Arnoldi with optional incomplete orthogonalization) | large sparse or matrix-free operators |
//! | [`matexp_leja`] | Leja interpolation with divided differences | large sparse or matrix-free operators |
//! | [`arnoldi`] | Arnoldi iteration used by the Krylov and Leja evaluators | |
//!
//! The evaluator interfaces are the traits
//! [`matexp_traits::DensePhikvEvaluator`] (dense $A$) and
//! [`matexp_traits::LinOpPhikvEvaluator`] (sparse or matrix-free $A$).
//!
//! ## Time integrators
//!
//! | Family | Methods | Module |
//! | ------ | ------- | ------ |
//! | Exponential propagation iterative | EPI2, EPI3 | [`ode_epirk`] |
//! | Exponential Rosenbrock | EXPRB2, EXPRB3 | [`ode_exprb`] |
//! | Explicit Runge-Kutta | RK1 (forward Euler), RK2, RK3, RK4 | [`ode_rk`] |
//! | Implicit | BDF1 (backward Euler), BDF2, Crank-Nicolson, SDIRK | [`ode_implicit`], [`tableau_implicit`] |
//!
//! Integrators are configured most easily with the builders in
//! [`integrator_builder`]. A problem is described by implementing
//! [`ode_sys::OdeSys`], and every integrator implements
//! [`ode_traits::IntegrateSys`]. A time step is proposed with `step` and
//! then committed with `accept_step`.
//!
//! ## Example: dense matrix exponential
//!
//! Compute $\exp(A t)$ for a small dense matrix with the Pade evaluator.
//!
//! ```
//! use ormatex::matexp_pade;
//!
//! // exp(dt * A) for a 2x2 matrix with eigenvalues -1 and -3
//! let a = faer::mat![[-2.0, 1.0], [1.0, -2.0]];
//! let dt = 0.5;
//! let e = matexp_pade::matexp(a.as_ref(), dt);
//!
//! // exact result: 1/2 * [[e^-0.5 + e^-1.5, e^-0.5 - e^-1.5], [.., ..]]
//! let (e1, e3) = ((-0.5_f64).exp(), (-1.5_f64).exp());
//! assert!((e[(0, 0)] - 0.5 * (e1 + e3)).abs() < 1e-12);
//! assert!((e[(0, 1)] - 0.5 * (e1 - e3)).abs() < 1e-12);
//! ```
//!
//! ## Example: exponential time integrator
//!
//! Integrate the linear decay chain $y^\prime  = A y$ with the second order EPI2
//! method, using a Krylov evaluator for the $\varphi$-function products.
//! A system supplies its right hand side $f(t, y)$ and a Jacobian
//! operator (here a finite difference Jacobian from [`ode_sys::get_fd_jac`]).
//!
//! ```
//! use faer::matrix_free::LinOp;
//! use faer::prelude::*;
//! use ormatex::integrator_builder::{
//!     ExponentialIntegratorBuilder, ExponentialMethod, KrylovOptions,
//! };
//! use ormatex::ode_sys::{get_fd_jac, OdeSys};
//!
//! /// y0' = -y0,  y1' = y0 - 2 y1
//! struct DecayChain;
//!
//! impl<'a> OdeSys<'a> for DecayChain {
//!     fn frhs(&self, _t: f64, y: MatRef<f64>) -> Mat<f64> {
//!         faer::mat![[-y[(0, 0)]], [y[(0, 0)] - 2.0 * y[(1, 0)]]]
//!     }
//!
//!     fn fjac<'b>(&'a self, t: f64, y: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
//!         Box::new(get_fd_jac(self, t, y))
//!     }
//! }
//!
//! let sys = DecayChain;
//! let y0 = faer::mat![[1.0], [0.0]];
//!
//! // build an EPI2 integrator with Krylov phi-function evaluation
//! let mut integrator =
//!     ExponentialIntegratorBuilder::new(0.0, y0.as_ref(), ExponentialMethod::Epi2)
//!         .with_krylov(KrylovOptions::default())
//!         .build()
//!         .unwrap();
//!
//! // propose a step, then accept it
//! let dt = 0.1;
//! for _ in 0..10 {
//!     let step = integrator.step(&sys, dt).unwrap();
//!     integrator.accept_step(step);
//! }
//!
//! // exact solution at t = 1: y0 = e^-1, y1 = e^-1 - e^-2
//! let y = integrator.state();
//! assert!((integrator.time() - 1.0).abs() < 1e-12);
//! assert!((y[(0, 0)] - (-1.0_f64).exp()).abs() < 1e-3);
//! ```
//!
//! Other exponential methods and evaluators are selected the same way, for
//! example `ExponentialMethod::Exprb3` with `.with_leja(LejaOptions::default())`.
//! Runnable programs are in the `examples/` directory of the repository.
//!
//! ## References
//!
//! * Hochbruck, M., Ostermann, A. Exponential integrators. Acta Numerica 19
//!   (2010) 209-286. doi:10.1017/S0962492910000048
//! * Higham, N. J. The scaling and squaring method for the matrix exponential
//!   revisited. SIAM J. Matrix Anal. Appl. 26(4) (2005) 1179-1193.
//!   doi:10.1137/04061101X
//! * Tokman, M. Efficient integration of large stiff systems of ODEs with
//!   exponential propagation iterative (EPI) methods. J. Comput. Phys. 213
//!   (2006) 748-776. doi:10.1016/j.jcp.2005.08.032
//! * Hochbruck, M., Ostermann, A., Schweitzer, J. Exponential Rosenbrock-type
//!   methods. SIAM J. Numer. Anal. 47(1) (2009) 786-803.
//!   doi:10.1137/080717717
//! * Gaudreault, S., Pudykiewicz, J. A. An efficient exponential time
//!   integration method for the numerical solution of the shallow water
//!   equations on the sphere. J. Comput. Phys. 322 (2016) 827-848.
//! * Gaudreault, S., Rainwater, G., Tokman, M. KIOPS: A fast adaptive Krylov
//!   subspace solver for exponential integrators. J. Comput. Phys. 372 (2018)
//!   236-255. doi:10.1016/j.jcp.2018.06.026
//! * Caliari, M., Cassini, F., Zivcovich, F. BAMPHI: Matrix-free and
//!   transpose-free action of linear combinations of phi-functions from
//!   exponential integrators. J. Comput. Appl. Math. 423 (2023) 114973.
//! * Pusa, M. Rational approximations to the matrix exponential in burnup
//!   calculations. Nucl. Sci. Eng. 169(2) (2011) 155-167.
//!   doi:10.13182/NSE10-81
//!
//! ## Citation
//!
//! If you find this software useful in your work, please cite: Gurecky,
//! William, and Pieper, Konstantin. ORMATEX. Computer Software.
//! <https://github.com/ORNL/ORMATEX>. USDOE. 24 Jan. 2025.
//! doi:10.11578/dc.20250124.7.
#![warn(missing_docs)]

pub mod arnoldi;
pub mod logger;
pub mod mat_utils;
pub mod matexp_cauchy;
pub mod matexp_krylov;
pub mod matexp_leja;
pub mod matexp_pade;
pub mod matexp_taylor;
pub mod matexp_traits;
pub mod newton;
pub mod ode_epirk;
pub mod ode_exprb;
pub mod ode_implicit;
pub mod ode_rk;
pub mod ode_sys;
pub mod ode_traits;
pub mod integrator_builder;
pub mod ode_step_controller;
pub mod tableau_implicit;

// for testing only
pub mod ode_utils;
pub mod test_common;

#[cfg(feature = "python")]
pub mod ormatex_rspy;
