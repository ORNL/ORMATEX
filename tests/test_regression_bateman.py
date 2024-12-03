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
    methods = ["epi2", "epi3", "exp2_dense", "exp3_dense"]
    for method in methods:
        t_res, y_res, t_true, y_true = analytic_bateman_s3(method=method, do_plot=False)
        diff = y_res - y_true
        print("Method: %s, Max abs err: %0.4e" % (method, np.max(np.abs(diff))))
        assert np.allclose(t_res, t_true)
        assert np.allclose(y_res, y_true, rtol=1e-9, atol=1e-9)
