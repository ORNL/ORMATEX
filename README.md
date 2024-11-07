# ORMATEX

**O**ak **R**idge **MAT**rix **EX**ponential tools.

ORMATEX contains methods to compute the matrix exponential:  $`\mathrm{exp}(A t)`$, and the action of the matrix exponential on a vector: $`\mathrm{exp}(A t)v_0`$, where $`A`$ is a matrix.  Additionally, this package contains related methods for the $`\varphi`$-functions.  Krylov methods to evaluate the matrix exponential-vector and $`\varphi`$-vector products are provided for cases where $`A`$ is large and sparse.
Utilizing these methods, ORMATEX implements performant exponential integrators for large systems of coupled ODEs.

ORMATEX is a mixed Rust and Python package that provides an extensible foundation to construct advanced exponential integrators.

The current set of implemented and planned time integration methods in each language:

### Rust

Exponential integrators:

- [x] EPIRK2  (exponential propagation iterative Runga-Kutta)
- [x] EPIRK3
- [ ] EPIRK4
- [ ] EXPROS4  (exponential rosenbrock order 4)
- [ ] EXPROS2

Classic integrators:

- [x] RK1,RK2,RK3,RK4
- [x] Backward Euler
- [x] BDF2
- [x] Crank-Nicolson
- [ ] DIRK
- [ ] SDIRK


### Python

Exponential integrators:

- [x] EPIRK2
- [x] EPIRK3
- [ ] EPIRK4
- [ ] EXPROS4
- [ ] EXPROS2

Classic integrators:

- [ ] RK1,RK2,RK3,RK4
- [ ] Backward Euler
- [ ] BDF2
- [ ] Crank-Nicolson
- [ ] DIRK
- [ ] SDIRK


# Python Setup

### Depends

- jax
- numpy
- scipy
- pytest
- python3.8+
- equinox
- matplotlib
- scikit-fem

### Install

For a local development install, run:

    pip install -e .

After running the above, the python unit tests can be executed.
From the project base directory (the directory this readme is located in), run:

    pytest

### Use

TODO

# Rust Setup

Download rustup: https://www.rust-lang.org/tools/install

Then, get rust dev stuff:

    rustup toolchain install stable

rustup handles installing the rust toolchain.
For improved editing, install the language server for rust (lsp), rust-analyzer:

    rustup component add rust-analyzer

To update rust toolchain

    rustup update

### Build

After setting up rust and cargo, to create a debug build run:

    cargo build

To run tests:

    cargo test

For an optimized build run:

    cargo build --release

### Examples

Run the examples with

    cargo run --example ex_sys_1 --release
    cargo run --example ex_sys_2 --release

Expected resulting images from running the first example of the Lotka-Volterra system integrated with EPIRK3:

![plot](./docs/images/ex_sys__ex_1.png)

Expected result from the Bateman system in the sectiond example integrated with EPIRK2:

![plot](./docs/images/ex_bateman.png)


License
========

TBD


References
==========

Incomplete orthogonalization procedure (IOM2), faster Arnoldi:

    [1] Saad, Yousef. "Variations on Arnoldi's method for computing eigenelements of large unsymmetric matrices." Linear algebra and its applications 34 (1980): 269-295.

Exponential (EPI) applied to shallow water equations:

    [2] Gaudreault, Stéphane, and Janusz A. Pudykiewicz.
    An efficient exponential time integration method for the numerical solution of the
    shallow water equations on the sphere.
    Journal of Computational Physics 322 (2016): 827-848.

Exponential Rosenbrock integrators applied to atmosphere and ocean:

    [3] Luan, Vu Thai, Janusz A. Pudykiewicz, and Daniel R. Reynolds. "Further development of efficient and accurate time integration schemes for meteorological models." Journal of Computational Physics 376 (2019): 817-837.
