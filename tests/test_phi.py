"""
Test phi-functions
"""
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
from ormatex_py.ode_sys import JaxMatrixLinop
from ormatex_py.matexp_phi import f_phi_k
from ormatex_py.matexp_krylov import phi_linop
jax.config.update("jax_enable_x64", True)


def test_phi_0():
    """
    Computes phi_0(A) == exp(A)
    """
    np.random.seed(42)
    for i in range(10):
        np_test_a = np.random.randn(10, 10)
        test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
        # test against the scipy variant (might be different impl?)
        np_phi_0 = sp.linalg.expm(np_test_a)
        k = 0
        jax_phi_0 = f_phi_k(test_a, k)
        jnp.allclose(jnp.array(np_phi_0), jax_phi_0)

def test_phi_1():
    """
    Computes phi_1(A)
    """
    np.random.seed(42)
    for i in range(10):
        np_test_a = np.random.randn(10, 10)
        test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
        k = 1
        jax_phi_0 = f_phi_k(test_a, k)
    pass

def test_phi_k():
    """
    Computes phi_k(A) with k >= 2
    """
    pass

def test_phi_linop_0():
    """
    Computes phi_0(A*dt)*b
    via krylov approx
    """
    np.random.seed(42)
    for i in range(10):
        np_test_a = np.random.randn(10, 10)
        test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
        b = jnp.ones((10,))
        # test against the scipy variant (might be different impl?)
        np_phi_0 = sp.linalg.expm(np_test_a)
        k = 0
        test_a_lo = JaxMatrixLinop(test_a)
        # do not use iom in this test
        jax_phi_0_b = phi_linop(
                test_a_lo, 1.0, b, k, max_krylov_dim=10, iom=100)
        jnp.allclose(jnp.array(np_phi_0), jax_phi_0_b)

def test_phi_linop_1():
    """
    Computes phi_1(A*dt)*b
    """
    pass
