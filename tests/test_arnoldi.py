"""
Unit tests for arnoldi
"""
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
from ormatex_py.arnoldi_jax import arnoldi_lop
from ormatex_py.ode_sys import MatrixLinOp
jax.config.update("jax_enable_x64", True)


def test_arnoldi_ortho():
    np.random.seed(42)
    for i in range(10):
        # construct a few random matricies
        test_a = jnp.asarray(np.random.randn(10, 10), dtype=jnp.float64)
        b = jnp.linspace(1, 10, 10, dtype=jnp.float64)
        # convert matrix to linear operator
        test_a_lop = MatrixLinOp(test_a)
        # call arnoldi iters for m iters
        m = 100
        # turn off incomplete ortho (use large ortho depth)
        iom = 10000
        # scale the linop (optional, typically 1.0)
        dt = 2.3
        qs, hs, bd = arnoldi_lop(test_a_lop, dt, b, m, iom)
        print("qs shape: ", qs.shape)
        print("hs shape: ", hs.shape)
        # since test_a is only 10x10, check that breakdown happened
        assert bd == 10
        # ensure orthonormal Q: Q^T*Q = I
        assert jnp.allclose(qs.T @ qs, jnp.eye(10))
        # ensure Q^T*A*Q = H
        assert jnp.allclose(dt * (qs.T @ test_a @ qs), hs)
