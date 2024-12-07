"""
Test phi-functions
"""
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp

from ormatex_py.ode_sys import MatrixLinOp
from ormatex_py.matexp_krylov import phi_linop, kiops_fixedsteps
from ormatex_py.matexp_phi import f_phi_k, f_phi_k_ext, f_phi_k_appl, f_phi_k_sq

jax.config.update("jax_enable_x64", True)

## reference grid and known values of the phi function for k=0,1,2,3
ref_z = np.array([-2., -1.2, -0.4, 0.4, 1.2, 2.])
ref_phi = np.array([
    [ 0.13533528323661267,  0.3011942119122021,   0.6703200460356393,
      1.4918246976412706,   3.3201169227365477,   7.389056098930651,  ],
    [ 0.43233235838169365,  0.5823381567398316,   0.8241998849109018,
      1.229561744103176,    1.933430768947123,    3.1945280494653256, ],
    [ 0.28383382080915315,  0.34805153605014033,  0.4395002877227457,
      0.5739043602579396,   0.7778589741226025,   1.0972640247326628, ],
    [ 0.10808308959542341,  0.1266237199582164,   0.15124928069313592,
      0.18476090064484876,  0.23154914510216867,  0.29863201236633136 ]])

def test_phi_0():
    """
    Computes phi_0(A) == exp(A)
    """

    # test a diagonal matrix with reference values
    d_z = jnp.diag(ref_z)
    jax_phi_0 = f_phi_k(d_z, k=0)
    assert jnp.allclose(jnp.diag(jax_phi_0), ref_phi[0,:])
    jax_phi_0 = f_phi_k_ext(d_z, k=0)
    assert jnp.allclose(jnp.diag(jax_phi_0), ref_phi[0,:])
    jax_phi_0 = f_phi_k_sq(d_z, k=0)
    assert jnp.allclose(jnp.diag(jax_phi_0), ref_phi[0,:])

    np.random.seed(42)
    for it in range(3):
        dim = 5*(it+1)
        np_test_a = np.random.randn(dim, dim)
        test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
        # test against scipy expm
        np_phi_0 = jnp.asarray(sp.linalg.expm(np_test_a))
        jax_phi_0 = f_phi_k(test_a, k=0)
        assert jnp.allclose(np_phi_0, jax_phi_0)
        jax_phi_0_ext = f_phi_k_ext(test_a, k=0)
        assert jnp.allclose(np_phi_0, jax_phi_0_ext)
        jax_phi_0_sq = f_phi_k_sq(test_a, k=0)
        assert jnp.allclose(np_phi_0, jax_phi_0_sq)

def run_test_phi_k(k):
    # test a diagonal matrix with reference values
    if k < ref_phi.shape[0]:
        d_z = jnp.diag(ref_z)
        jax_phi_k = f_phi_k(d_z, k=k)
        assert jnp.allclose(jnp.diag(jax_phi_k), ref_phi[k,:])
        jax_phi_k = f_phi_k_ext(d_z, k=k)
        assert jnp.allclose(jnp.diag(jax_phi_k), ref_phi[k,:])
        jax_phi_k = f_phi_k_sq(d_z, k=k)
        assert jnp.allclose(jnp.diag(jax_phi_k), ref_phi[k,:])

    np.random.seed(42)
    for it in range(3):
        dim = 5*(it+1)
        np_test_a = np.random.randn(dim, dim)
        test_a = jnp.asarray(np_test_a, dtype=jnp.float64)

        # test against different impl.
        jax_phi_k = f_phi_k(test_a, k=k)
        jax_phi_k_ext = f_phi_k_ext(test_a, k=k)
        assert jnp.allclose(jax_phi_k, jax_phi_k_ext)
        jax_phi_k_sq = f_phi_k_sq(test_a, k=k)
        #jax.debug.print("{a}\n{b}", a=jax_phi_k, b=jax_phi_k_poly)
        assert jnp.allclose(jax_phi_k_sq, jax_phi_k_ext)

def test_phi_1():
    """
    Computes phi_1(A)
    """
    run_test_phi_k(1)

def test_phi_k_seqential():
    """
    Computes phi_k(A) with k >= 2
    """
    for k in range(2, ref_phi.shape[0]):
        run_test_phi_k(k)

def test_phi_k_all():
    """
    test routines that compute phi_k(A) for all k < k_max
    """

    k_max = ref_phi.shape[0]-1
    d_z = jnp.diag(ref_z)
    jax_phi_ks = f_phi_k_ext(d_z, k=k_max, return_all=True)
    diags = lambda z: jnp.einsum("kii -> ki", z)
    assert jnp.allclose(diags(jax_phi_ks), ref_phi)
    jax_phi_ks = f_phi_k_sq(d_z, k=k_max, return_all=True)
    jax.debug.print("{a}\n{b}", a=diags(jax_phi_ks), b=ref_phi)
    assert jnp.allclose(diags(jax_phi_ks), ref_phi)

    #np.random.seed(42)
    #for it in range(3):
    #    dim = 5*(it+1)
    #    np_test_a = np.random.randn(dim, dim)
    #    test_a = jnp.asarray(np_test_a, dtype=jnp.float64)

    #    # test against different impl.
    #    jax_phi_k = f_phi_k(test_a, k=k)
    #    jax_phi_k_ext = f_phi_k_ext(test_a, k=k)
    #    assert jnp.allclose(jax_phi_k, jax_phi_k_ext)
    #    jax_phi_k_sq = f_phi_k_sq(test_a, k=k)
    #    #jax.debug.print("{a}\n{b}", a=jax_phi_k, b=jax_phi_k_poly)
    #    assert jnp.allclose(jax_phi_k_sq, jax_phi_k_ext)


def test_phi_k_appl():
    """
    Computes phi_k(A)B with k >= 0
    """
    for k in range(ref_phi.shape[0]):
        # test a diagonal matrix with reference values
        d_z = jnp.diag(ref_z)
        b = jnp.ones((ref_phi.shape[1],))
        jax_phi_k_b = f_phi_k_appl(d_z, b, k=k)
        assert jnp.allclose(jax_phi_k_b, ref_phi[k,:])

        np.random.seed(42)
        for it in range(3):
            dim = 5*(it+1)
            np_test_a = np.random.randn(dim, dim)
            np_test_b = np.random.randn(dim, 1)
            test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
            test_b = jnp.asarray(np_test_b, dtype=jnp.float64)

            # test against different impl.
            jax_phi_k = f_phi_k(test_a, k=k)
            jax_phi_k_b = f_phi_k_appl(test_a, test_b, k=k)
            assert jnp.allclose(jax_phi_k @ test_b, jax_phi_k_b)

def test_phi_linop_0():
    """
    Computes phi_0(A*dt)*b
    via Krylov approx
    """

    # test a diagonal matrix with reference values
    d_z = np.diag(ref_z)
    d_z_lo = MatrixLinOp(jnp.asarray(d_z))
    b = jnp.ones((ref_z.shape[0],))
    jax_phi_0_b = phi_linop(
        d_z_lo, 1.0, b, k=0, max_krylov_dim=ref_z.shape[0], iom=100)
    assert jnp.allclose(jax_phi_0_b, ref_phi[0,:])

    np.random.seed(42)
    for it in range(10):
        dim = it+1
        np_test_a = np.random.randn(dim, dim)
        test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
        b = jnp.ones((dim,))
        # test against the scipy expm
        np_phi_0 = jnp.asarray(sp.linalg.expm(np_test_a))
        test_a_lo = MatrixLinOp(test_a)
        # do not use iom in this test
        jax_phi_0_b = phi_linop(
                test_a_lo, 1.0, b, k=0, max_krylov_dim=dim, iom=100)

        assert jnp.allclose(np_phi_0 @ b, jax_phi_0_b)

def test_phi_linop_1():
    """
    Computes phi_1(A*dt)*b
    via Krylov approx
    """

    # test a diagonal matrix with reference values
    d_z = np.diag(ref_z)
    d_z_lo = MatrixLinOp(jnp.asarray(d_z))
    b = jnp.ones((ref_z.shape[0],))
    jax_phi_1_b = phi_linop(
        d_z_lo, 1.0, b, k=1, max_krylov_dim=ref_z.shape[0], iom=100)
    assert jnp.allclose(jax_phi_1_b, ref_phi[1,:])

def test_kiops_fixedstep():
    """
    Test that the KIOPS method correctly computes
    linear combinations of phi-vector products:

    kiops_w = phi_0(dt*A)*0 + phi_1*(dt*A)*(dt)*b1 + phi_2*(dt*A)*(dt)*b2
    """
    # === small bateman test case for phipm
    n = 3
    np_test_a = np.array([
        [-1e-3, 1.e-1, 0.0],
        [   0.,-1.e-1, 1e1],
        [   0.,    0.,-1e1],
        ])
    test_a = jnp.asarray(np_test_a, dtype=jnp.float64)
    test_a_lo = MatrixLinOp(test_a)

    # arbitrary small vectors with vastly different magnitudes
    test_b0 = jnp.asarray(np.zeros(n))
    test_b1 = jnp.asarray(np.linspace(0.2, 3.4, n) * 1e-1)
    test_b2 = jnp.asarray(np.linspace(0.8, 4.0, n) * 1e-4)
    dt = 2.5

    base_phi_1_b1 = f_phi_k_ext(dt*test_a, 1) @ test_b1
    base_phi_2_b2 = f_phi_k_ext(dt*test_a, 2) @ test_b2
    base_phi_combo = base_phi_1_b1 + base_phi_2_b2

    # compute same thing using phipm method
    vb = [test_b0, test_b1, test_b2]
    phipm_phi_combo = kiops_fixedsteps(test_a_lo, dt, vb, max_krylov_dim=10, iom=10, n_steps=1)
    assert jnp.allclose(phipm_phi_combo, base_phi_combo, atol=1e-6)
