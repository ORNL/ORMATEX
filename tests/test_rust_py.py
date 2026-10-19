"""
Test the rust python ORMATEX bindings
"""
import numpy as np
import pytest
import jax
from jax import numpy as jnp
from ormatex_py.integrate_wrapper import integrate
from ormatex_py.ode_sys import MatrixLinOp
from ormatex_py.arnoldi_jax import arnoldi_lop
from ormatex_py.progression.lotka_volterra import LotkaVolterra
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


@pytest.mark.skipif(not HAS_ORMATEX_RUST, reason="rust ormatex bindings not built")
def test_integrate_rs_krylov_max_substeps():
    """
    Krylov substepping (max_substeps) passed through the rust bindings agrees
    with the unsubstepped solution.
    """
    sys = LotkaVolterra()
    y0 = jnp.array([0.1, 0.2])
    kwargs = dict(phi_method="krylov", tol=1e-12, max_krylov_dim=20, iom=20)
    res0 = integrate(sys, y0, 0.0, 0.05, 20, "epi2_rs", **kwargs)
    for max_substeps in [2, 4]:
        res = integrate(sys, y0, 0.0, 0.05, 20, "epi2_rs",
                        max_substeps=max_substeps, **kwargs)
        assert np.allclose(np.asarray(res.y), np.asarray(res0.y), atol=1e-9)


@pytest.mark.skipif(not HAS_ORMATEX_RUST, reason="rust ormatex bindings not built")
def test_integrate_rs_krylov_rejects_zero_iom():
    """
    The incomplete orthogonalization depth must be at least 1.
    """
    sys = LotkaVolterra()
    y0 = jnp.array([0.1, 0.2])
    with pytest.raises(ValueError):
        integrate(sys, y0, 0.0, 0.05, 2, "epi2_rs", phi_method="krylov", iom=0)
