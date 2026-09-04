"""
Regression test for pure linear Bateman system.

Check that all exponential integrators solve a simple
Bateman system to the expected precission.
"""
from ormatex_py.progression.bateman_sys import analytic_bateman_s3
import numpy as np
import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)


def test_bateman_s3():
    """
    Test exponential integrators for a 3 species system
    """
    cases = [
        ("epi2", "dense"),
        ("epi3", "krylov"),
        ("exprb3", "dense"),
        ("pexprb4", "pfd"),
        ("exp3_dense", ""),
        ("epi3", "pfd"),
    ]
    for method, phi_method in cases:
        t_res, y_res, t_true, y_true = analytic_bateman_s3(method=method, phi_method=phi_method, do_plot=False)
        diff = y_res - y_true
        print("Method: %s<%s>, Max abs err: %0.4e" % (method, phi_method, np.max(np.abs(diff))))
        assert np.allclose(t_res, t_true)
        assert np.allclose(y_res, y_true, rtol=1e-9, atol=1e-9)


def test_bateman_s3_pfd():
    """
    Test exponential integrators for a 3 species system
    """
    cases = [
        ("exprb2", "pfd", "cram_6"),
        ("exprb2", "pfd", "cram_16"),
        ("exprb2", "pfd", "pade_7_8"),
    ]
    for method, phi_method, pfd_method in cases:
        t_res, y_res, t_true, y_true = analytic_bateman_s3(
            method=method, phi_method=phi_method, do_plot=False, pfd_method=pfd_method)
        diff = y_res - y_true
        print("Method: %s<%s> %s, Max abs err: %0.4e" % (method, phi_method, pfd_method, np.max(np.abs(diff))))
        assert np.allclose(t_res, t_true)
        assert np.allclose(y_res, y_true, rtol=1e-3, atol=1e-3)
