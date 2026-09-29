# ORMATEX

**O**ak **R**idge **MAT**rix **EX**ponential tools.

ORMATEX computes the matrix exponential `exp(A t)`, its action on a vector
`exp(A t) v0`, and the related phi-functions

```text
phi_0(z) = exp(z),    phi_{k+1}(z) = (phi_k(z) - 1/k!) / z
```

For small dense matrices these are evaluated directly. For large and sparse (or
matrix-free) operators `A`, Krylov subspace and Leja polynomial methods evaluate
the vector products `phi_k(t A) v` using only matrix-vector products with `A`.

On top of these kernels the crate provides exponential time integrators for
large systems of coupled ODEs

```text
dy/dt = f(t, y),    y(t0) = y0
```

together with classic explicit and implicit Runge-Kutta and BDF integrators for
comparison. The linear algebra is built on [faer](https://docs.rs/faer/latest/faer/).

ORMATEX is a mixed Rust and Python package. The Rust integrators are also
available from Python through the `ormatex_rspy` module (cargo feature `python`,
enabled by default). The Python interface, which additionally offers JAX based
integrators, is described in the
[project README](https://github.com/ORNL/ORMATEX#readme).

Full API documentation, with rendered math, is on
[docs.rs](https://docs.rs/ormatex).

## Matrix exponential and phi-function evaluators

| Module | Method | Best suited for |
| ------ | ------ | --------------- |
| [`matexp_pade`](https://docs.rs/ormatex/latest/ormatex/matexp_pade/) | Pade approximation with scaling and squaring (Higham) | small dense matrices, real or complex |
| [`matexp_taylor`](https://docs.rs/ormatex/latest/ormatex/matexp_taylor/) | Taylor series | small dense matrices, real or complex |
| [`matexp_cauchy`](https://docs.rs/ormatex/latest/ormatex/matexp_cauchy/) | Contour integral (CRAM, parabolic contour) via partial fractions | real dense matrices with spectrum near the negative real axis |
| [`matexp_krylov`](https://docs.rs/ormatex/latest/ormatex/matexp_krylov/) | Krylov subspace (Arnoldi with optional incomplete orthogonalization) | large sparse or matrix-free operators |
| [`matexp_leja`](https://docs.rs/ormatex/latest/ormatex/matexp_leja/) | Leja interpolation with divided differences | large sparse or matrix-free operators |

The evaluator interfaces are the traits
[`DensePhikvEvaluator`](https://docs.rs/ormatex/latest/ormatex/matexp_traits/trait.DensePhikvEvaluator.html)
(dense `A`) and
[`LinOpPhikvEvaluator`](https://docs.rs/ormatex/latest/ormatex/matexp_traits/trait.LinOpPhikvEvaluator.html)
(sparse or matrix-free `A`).

## Time integrators

| Family | Methods | Module |
| ------ | ------- | ------ |
| Exponential propagation iterative | EPI2, EPI3 | [`ode_epirk`](https://docs.rs/ormatex/latest/ormatex/ode_epirk/) |
| Exponential Rosenbrock | EXPRB2, EXPRB3 | [`ode_exprb`](https://docs.rs/ormatex/latest/ormatex/ode_exprb/) |
| Explicit Runge-Kutta | RK1 (forward Euler), RK2, RK3, RK4 | [`ode_rk`](https://docs.rs/ormatex/latest/ormatex/ode_rk/) |
| Implicit | BDF1 (backward Euler), BDF2, Crank-Nicolson, SDIRK | [`ode_implicit`](https://docs.rs/ormatex/latest/ormatex/ode_implicit/), [`tableau_implicit`](https://docs.rs/ormatex/latest/ormatex/tableau_implicit/) |

Integrators are configured most easily with the builders in
[`integrator_builder`](https://docs.rs/ormatex/latest/ormatex/integrator_builder/).
A problem is described by implementing
[`OdeSys`](https://docs.rs/ormatex/latest/ormatex/ode_sys/trait.OdeSys.html),
and every integrator implements
[`IntegrateSys`](https://docs.rs/ormatex/latest/ormatex/ode_traits/trait.IntegrateSys.html).
A time step is proposed with `step` and then committed with `accept_step`.

## Example: dense matrix exponential

Compute `exp(A t)` for a small dense matrix with the Pade evaluator.

```rust
use ormatex::matexp_pade;

// exp(dt * A) for a 2x2 matrix with eigenvalues -1 and -3
let a = faer::mat![[-2.0, 1.0], [1.0, -2.0]];
let dt = 0.5;
let e = matexp_pade::matexp(a.as_ref(), dt);

// exact result: 1/2 * [[e^-0.5 + e^-1.5, e^-0.5 - e^-1.5], [.., ..]]
let (e1, e3) = ((-0.5_f64).exp(), (-1.5_f64).exp());
assert!((e[(0, 0)] - 0.5 * (e1 + e3)).abs() < 1e-12);
assert!((e[(0, 1)] - 0.5 * (e1 - e3)).abs() < 1e-12);
```

## Example: exponential time integrator

Integrate the linear system `y' = A y` with the second order EPI2 method, using a
Krylov evaluator for the phi-function products. A system supplies its right hand
side `f(t, y)` and a Jacobian operator (here a finite difference Jacobian from
`get_fd_jac`).

```rust
use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::integrator_builder::{
    ExponentialIntegratorBuilder, ExponentialMethod, KrylovOptions,
};
use ormatex::ode_sys::{get_fd_jac, OdeSys};

/// y0' = -y0,  y1' = y0 - 2 y1
struct LinearSys;

impl<'a> OdeSys<'a> for LinearSys {
    fn frhs(&self, _t: f64, y: MatRef<f64>) -> Mat<f64> {
        faer::mat![[-y[(0, 0)]], [y[(0, 0)] - 2.0 * y[(1, 0)]]]
    }

    fn fjac<'b>(&'a self, t: f64, y: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(get_fd_jac(self, t, y))
    }
}

let sys = LinearSys;
let y0 = faer::mat![[1.0], [0.0]];

// build an EPI2 integrator with Krylov phi-function evaluation
let mut integrator =
    ExponentialIntegratorBuilder::new(0.0, y0.as_ref(), ExponentialMethod::Epi2)
        .with_krylov(KrylovOptions::default())
        .build()
        .unwrap();

// propose a step, then accept it
let dt = 0.1;
for _ in 0..10 {
    let step = integrator.step(&sys, dt).unwrap();
    integrator.accept_step(step);
}

// exact solution at t = 1: y0 = e^-1, y1 = e^-1 - e^-2
let y = integrator.state();
assert!((integrator.time() - 1.0).abs() < 1e-12);
assert!((y[(0, 0)] - (-1.0_f64).exp()).abs() < 1e-3);
```

Other exponential methods and evaluators are selected the same way, for example
`ExponentialMethod::Exprb3` with `.with_leja(LejaOptions::default())`. Runnable
programs are in the
[`examples/`](https://github.com/ORNL/ORMATEX/tree/main/examples) directory of
the repository.

## References

* Kazdadi, S. Q. E., (2026). faer: A linear algebra library for the Rust
  programming language. Journal of Open Source Software, 11(123), 6099,
  <https://doi.org/10.21105/joss.06099>
* Hochbruck, M., Ostermann, A. Exponential integrators. Acta Numerica 19
  (2010) 209-286. doi:10.1017/S0962492910000048
* Higham, N. J. The scaling and squaring method for the matrix exponential
  revisited. SIAM J. Matrix Anal. Appl. 26(4) (2005) 1179-1193.
  doi:10.1137/04061101X
* Tokman, M. Efficient integration of large stiff systems of ODEs with
  exponential propagation iterative (EPI) methods. J. Comput. Phys. 213
  (2006) 748-776. doi:10.1016/j.jcp.2005.08.032
* Hochbruck, M., Ostermann, A., Schweitzer, J. Exponential Rosenbrock-type
  methods. SIAM J. Numer. Anal. 47(1) (2009) 786-803.
  doi:10.1137/080717717
* Gaudreault, S., Pudykiewicz, J. A. An efficient exponential time
  integration method for the numerical solution of the shallow water
  equations on the sphere. J. Comput. Phys. 322 (2016) 827-848.
* Gaudreault, S., Rainwater, G., Tokman, M. KIOPS: A fast adaptive Krylov
  subspace solver for exponential integrators. J. Comput. Phys. 372 (2018)
  236-255. doi:10.1016/j.jcp.2018.06.026
* Caliari, M., Cassini, F., Zivcovich, F. BAMPHI: Matrix-free and
  transpose-free action of linear combinations of phi-functions from
  exponential integrators. J. Comput. Appl. Math. 423 (2023) 114973.
* Pusa, M. Rational approximations to the matrix exponential in burnup
  calculations. Nucl. Sci. Eng. 169(2) (2011) 155-167.
  doi:10.13182/NSE10-81

## Citation

If you find this software useful in your work, please cite:

* Gurecky, William, and Pieper, Konstantin. ORMATEX. Computer Software.
  <https://github.com/ORNL/ORMATEX>. USDOE. 24 Jan. 2025.
  [doi:10.11578/dc.20250124.7](https://doi.org/10.11578/dc.20250124.7).

## License

Copyright (c) 2024-present, UT-Battelle, LLC. Licensed under the Apache License,
Version 2.0.
