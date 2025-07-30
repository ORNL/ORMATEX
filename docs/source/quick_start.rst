Install Guide
=============

Python Only Install
-------------------

To obtain the latest development version of ORMATEX, clone the repo with:

.. code::

    git clone https://github.com/ORNL/ORMATEX.git

For a local development install, run:

.. code::

    cd ORMATEX
    pip install -e .

After running the above, the python unit tests can be executed.
From the project base directory (the directory this readme is located in), run:

.. code::

    pytest

Rust and Python Bindings
------------------------

Download rustup: https://www.rust-lang.org/tools/install

Then install the rust toolchain:

.. code::

    rustup toolchain install stable

After setting up rust and cargo, for an optimized build run:

.. code::

    cargo build --release

To use the rust-based ORMATEX integrators from a python interface, build the python-rust bindings with:

.. code::

    pip install maturin
    maturin develop --release

Ensure to use the ``--release`` flag for an optimized build.  Forgetting this flag will build in debug mode and will result in significantly degraded performance.


Quick Start
===========

Imports

.. code::

    import jax
    from jax import numpy as jnp
    from ormatex_py.ode_sys import OdeSys, CustomJacLinOp
    from ormatex_py import integrate_wrapper

Define the system

.. code::

    class LotkaVolterraAD(OdeSys):
        alpha: float
        beta: float
        delta: float
        gamma: float

        def __init__(self, a=1.0, b=1.0, d=1.0, g=1.0, **kwargs):
            super().__init__()
            self.alpha = a
            self.beta = b
            self.delta = d
            self.gamma = g

        @jax.jit
        def _frhs(self, t, x, **kwargs):
            prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
            pred_t = self.delta * x[0] * x[1] - self.gamma * x[1]
            return jnp.array([prey_t, pred_t])

Initialize the system and integrate

.. code::

    method = 'epi3'
    sys = LotkaVolterraAD()
    y0 = jnp.array([0.1, 0.2])
    t0 = 0.0
    dt = 0.2
    nsteps = 100
    res = integrate_wrapper.integrate(sys, y0, t0, dt, nsteps, method, max_krylov_dim=4, iom=2)
    t_res, y_res = res.t, res.y

Optionally, an explicit Jacobian can be supplied.  If not supplied, as above, automatic differentiation will be used.

.. code::

    class LotkaVolterra(OdeSys):
        alpha: float
        beta: float
        delta: float
        gamma: float

        def __init__(self, a=1.0, b=1.0, d=1.0, g=1.0, **kwargs):
            super().__init__()
            self.alpha = a
            self.beta = b
            self.delta = d
            self.gamma = g

        @jax.jit
        def _frhs(self, t, x, **kwargs):
            prey_t = self.alpha * x[0] - self.beta * x[0] * x[1]
            pred_t = self.delta * x[0] * x[1] - self.gamma * x[1]
            return jnp.array([prey_t, pred_t])

        @jax.jit
        def _fjac(self, t, x, **kwargs):
            jac = jnp.array([
                [self.alpha - self.beta * x[1], - self.beta*x[0]],
                [self.delta*x[1], self.delta*x[0] - self.gamma]
                ])
            return CustomJacLinOp(t, x, self.frhs, jac)

