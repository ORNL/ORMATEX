"""
Test phi-functions
"""
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
from ormatex_py.ode_sys import JaxMatrixLinop
from ormatex_py.matexp_phi import f_phi_k
from ormatex_py.matexp_krylov import phi_linop, phipm_unstable
jax.config.update("jax_enable_x64", True)


def test_phipm_unstable():
    """
    Test that the phipm correctly computes linear combinations
    of phi-vector products:

    phipm
    ==
    phi_1*(dt*A)*(dt)*b1 +
    phi_2*(dt*A)*(dt)*b2

    where
    phipm = phi_0(dt*A)*0 + phi_1*(dt*A)*(dt)*b1 + phi_2*(dt*A)*(dt)*b2
    """
    # make a random A matrix
    n = 10
    np.random.seed(42)
    np_test_a = np.random.randn(n, n)
    test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
    test_a_lo = JaxMatrixLinop(test_a)

    # arbitrary vectors
    test_b0 = jnp.asarray(np.zeros(n))
    test_b1 = jnp.asarray(np.linspace(0.2, 3.4, n) * 2.0)
    test_b2 = jnp.asarray(np.linspace(0.8, 4.0, n) * 1.0)
    # some random stepsize
    dt = 5.0

    base_phi_1_b1 = f_phi_k(dt*test_a, 1) @ test_b1
    base_phi_2_b2 = f_phi_k(dt*test_a, 2) @ test_b2
    base_phi_combo = base_phi_1_b1 + base_phi_2_b2

    # compute same thing using phipm method
    vb = [test_b0, test_b1, test_b2]
    p = 2
    phipm_phi_combo = phipm_unstable(test_a_lo, dt, vb, p, max_krylov_dim=n, iom=20)
    # assert jnp.allclose(phipm_phi_combo, base_phi_combo)

    # === small bateman test case for phipm
    n = 3
    np_test_a = np.array([
        [-1e-3, 1.e-1, 0.0],
        [   0.,-1.e-1, 1e1],
        [   0.,    0.,-1e1],
        ])
    test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
    test_a_lo = JaxMatrixLinop(test_a)

    # arbitrary small vectors with vastly different magnitudes
    test_b0 = jnp.asarray(np.zeros(n))
    test_b1 = jnp.asarray(np.linspace(0.2, 3.4, n) * 1e-1)
    test_b2 = jnp.asarray(np.linspace(0.8, 4.0, n) * 1e-9)
    dt = 5.0

    base_phi_1_b1 = f_phi_k(dt*test_a, 1) @ test_b1
    base_phi_2_b2 = f_phi_k(dt*test_a, 2) @ test_b2
    base_phi_combo = base_phi_1_b1 + base_phi_2_b2

    # compute same thing using phipm method
    vb = [test_b0, test_b1, test_b2]
    p = 2
    phipm_phi_combo = phipm_unstable(test_a_lo, dt, vb, p, max_krylov_dim=n, iom=20)
    assert jnp.allclose(phipm_phi_combo, base_phi_combo)


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
