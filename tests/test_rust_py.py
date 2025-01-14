"""
Test the rust python ORMATEX bindings
"""
import numpy as np
import jax
from jax import numpy as jnp
from ormatex_py.integrate_wrapper import integrate
from ormatex_py.ode_sys import MatrixLinOp
from ormatex_py.arnoldi_jax import arnoldi_lop
jax.config.update("jax_enable_x64", True)

try:
    from ormatex_py.ormatex import arnoldi_rs, integrate_wrapper_rs
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False


def test_arnoldi_rs():
    """
    Test the rust based arnoldi impl
    """
    if HAS_ORMATEX_RUST:
        np.random.seed(42)
        # generate a random matrix
        d = 5
        a = np.random.normal(0, 1., size=(d, d))
        a_lo = MatrixLinOp(a)
        b = np.random.normal(0, 1., size=(d, 1))

        # run jax arnoldi
        iom = 1000
        q_py, h_py = arnoldi_lop(a_lo, 1.0, b, 1000, iom)

        # run rust arnoldi
        dt = 1.0
        q_rs, h_rs, bkdwn_rs = arnoldi_rs(a_lo, dt, b, 1000, iom)

        # check ortho
        # ensure orthonormal Q: Q^T*Q = I
        assert np.allclose(q_rs.T @ q_rs, np.eye(d))
        # ensure Q^T*A*Q = H
        assert np.allclose(dt * (q_rs.T @ a @ q_rs), h_rs)

        # check result against jax python impl for same matrix
        assert np.allclose(q_rs, q_py)
        assert np.allclose(h_rs, h_py)

        # incomplete ortho check
        iom = 2
        q_py, h_py = arnoldi_lop(a_lo, 1.0, b, 8, iom)
        q_rs, h_rs, bkdwn_rs = arnoldi_rs(a_lo, dt, b, 8, iom)

        # check result against jax python impl for same matrix
        assert np.allclose(q_rs, q_py)
        assert np.allclose(h_rs, h_py)
