"""
Tests divided difference calculations
for Newton polynomial based
approximation of matexp-vector products.
"""
import numpy  as np
from copy import deepcopy
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from ormatex_py.matexp_leja import leja_coeffs_dd_phi, leja_coeffs_exp, leja_coeffs_exp_dd, leja_coeffs_ts_phi, leja_coeffs_ts_phi_jax, leja_seq_reordering, leja_coeffs_ts_ss, dd_exp_taylor, leja_seq_pad_zeros

from ormatex_py.matexp_leja import leja_shift_scale, gen_complex_conj_leja_fast, gen_leja_circle, gen_leja_fast, gen_leja_conjugate, newton_poly_div_diff

# arbitrary precision arithmitic package
from mpmath import mp


def newton_poly_div_diff_mp(x, y, dps=80):
    r"""
    NOTE: For testing only.

    Args:
        x: data support points, where the data is sampled.
        y: value of the function at x points.
        dps: digits of prescision
    """
    mp.dps = dps
    m = len(x)
    # high precision complex arrays
    a = mp.matrix([mp.mpc(complex(y[i])) for i in range(m)])
    xi = mp.matrix([mp.mpc(complex(x[i])) for i in range(m)])
    for k in range(1, m):
        for b in range(k, m):
            a[b] = (a[b] - a[k-1]) / (xi[b] - xi[k-1])
    return np.asarray([complex(v) for v in a])


def _conv_allclose(a, b, rtol=1e-4):
    # convert jax arrays to np then compare
    return np.allclose(np.asarray(a, np.complex64),
                       np.asarray(b, np.complex64), rtol=rtol)


def test_dd_exp_re():
    r"""
    tests the computation of the divided differences of the
    fuction

    .. math:

        exp(\Xi), \Xi \in [-2, 2]

    at the first 10 leja points
    """
    # generate real leja points
    leja_x = gen_leja_fast(-2.0, 2.0, 10)

    # high precision naive method recursive divided diff formula
    coeffs_mp = newton_poly_div_diff_mp(leja_x, jnp.exp(leja_x))

    # naive method recursive divided diff formula
    coeffs_dd = newton_poly_div_diff(leja_x, jnp.exp(leja_x))
    assert _conv_allclose(coeffs_mp, coeffs_dd)

    # matrix exp approximation to the divided differences
    coeffs_exp = leja_coeffs_exp(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_exp)

    # ORMATEX taylor series with scaling and squaring
    coeffs_ts = dd_exp_taylor(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_exp, coeffs_ts)

    # Taylor Series (II) method
    coeffs_tsii = leja_coeffs_ts_ss(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_exp, coeffs_tsii)

    # dd_phi method
    coeffs_dd_phi = leja_coeffs_dd_phi(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_exp, coeffs_dd_phi)


def test_dd_exp_re_30():
    """ Test 30 real leja points """
    leja_x = gen_leja_fast(-2.0, 2.0, 30)
    coeffs_mp = newton_poly_div_diff_mp(leja_x, jnp.exp(leja_x))
    coeffs_dd = newton_poly_div_diff(leja_x, jnp.exp(leja_x))
    coeffs_exp = leja_coeffs_exp(leja_x, 0.0, 1.0)
    coeffs_ts = dd_exp_taylor(leja_x, 0.0, 1.0)
    coeffs_dd_phi = leja_coeffs_dd_phi(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_exp)
    assert _conv_allclose(coeffs_mp, coeffs_ts)
    assert _conv_allclose(coeffs_mp, coeffs_dd_phi)


def test_dd_exp_im():
    r"""
    tests the computation of the divided differences of the
    fuction

    .. math:

        exp(\Xi), \Xi \in [-2i, 2i]

    for the leja points on the imaginary axis
    """
    leja_x = gen_complex_conj_leja_fast(-2.0, 10)
    coeffs_mp = newton_poly_div_diff_mp(leja_x, jnp.exp(leja_x))

    coeffs_dd = newton_poly_div_diff(leja_x, jnp.exp(leja_x))
    assert _conv_allclose(coeffs_mp, coeffs_dd)

    coeffs_exp = leja_coeffs_exp(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_exp)

    coeffs_tsii = leja_coeffs_ts_phi(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_tsii)

    coeffs_dd_phi = leja_coeffs_dd_phi(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_dd_phi)

def test_dd_exp_ellipse():
    r"""
    Test div diffs of

    .. math:

        exp(\Xi)

    with $`\Xi`$ lie on an ellipse in the complex plane
    """
    leja_x = gen_leja_conjugate(10, -2.0, 2.0, 2.0)[0]
    coeffs_mp = newton_poly_div_diff_mp(leja_x, jnp.exp(leja_x))
    coeffs_dd = newton_poly_div_diff(leja_x, jnp.exp(leja_x))
    assert _conv_allclose(coeffs_mp, coeffs_dd)
    coeffs_exp = leja_coeffs_exp(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_exp)
    coeffs_tsii = leja_coeffs_ts_phi(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_tsii)
    coeffs_dd_phi = leja_coeffs_dd_phi(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_mp, coeffs_dd_phi)
    coeffs_ts = dd_exp_taylor(leja_x, 0.0, 1.0)
    assert _conv_allclose(coeffs_exp, coeffs_ts, rtol=1e-6)


def test_dd_exp_shift_scale():
    r"""
    tests the computation of the divided differences of the
    shifted and scaled function:

    .. math:

        exp( a+ b\Xi), \Xi \in [-2, 2]

    """
    # generate real leja points
    leja_x = gen_leja_fast(-2.0, 2.0, 80)
    alpha = -101.0
    beta = 0
    shift = (alpha - beta) / 2.
    scale = abs(beta - alpha) / 4.

    # high precision naive method recursive divided diff formula
    coeffs_mp = newton_poly_div_diff_mp(leja_x, jnp.exp(shift + scale*leja_x))
    coeffs_exp = leja_coeffs_exp(leja_x, shift, scale)
    coeffs_ts = dd_exp_taylor(leja_x, shift, scale)
    # coeffs_tsss = leja_coeffs_ts_ss(leja_x, shift, scale)
    # coeffs_ts_phi = leja_coeffs_ts_phi(leja_x, shift, scale)
    assert _conv_allclose(coeffs_exp, coeffs_ts)


def test_dd_exp_shift_scale_substep():
    r"""
    Tests divided differences with substepping
    """
    # generate real leja points
    leja_x = gen_leja_fast(-2.0, 2.0, 80)
    alpha = -101.0
    beta = 0
    shift = (alpha - beta) / 2.
    scale = abs(beta - alpha) / 4.
    # substep size
    h = 0.25

    coeffs_exp = leja_coeffs_exp(leja_x, shift, scale, h)
    coeffs_ts = dd_exp_taylor(leja_x, shift, scale, h)
    assert _conv_allclose(coeffs_exp, coeffs_ts, rtol=1e-6)


def test_dd_exp_ellipse_pad_zeros():
    r"""
    Tests divided differences of a zero-padded leja sequence
    """
    leja_x = gen_leja_conjugate(80, -2.0, 2.0, 2.0)[0]
    zleja_x = leja_seq_pad_zeros(leja_x, 2)
    alpha = -101.0
    beta = 0
    shift = (alpha - beta) / 2.
    scale = abs(beta - alpha) / 4.
    # substep size
    h = 0.5

    coeffs_exp = leja_coeffs_exp(zleja_x, shift, scale, h)
    coeffs_ts = dd_exp_taylor(zleja_x, shift, scale, h)
    coeffs_ts_2 = dd_exp_taylor(zleja_x, shift, scale, h, mu_scale=0.0)
    assert _conv_allclose(coeffs_exp, coeffs_ts, rtol=1e-6)


def test_dd_exp_shift_scale_all():
    r"""
    tests the computation of the divided differences of the
    shifted and scaled exp function.
    """
    # generate real leja points
    leja_x = gen_leja_fast(-2.0, 2.0, 10)
    alpha = -11.0
    beta = 0
    shift = (alpha - beta) / 2.
    scale = abs(beta - alpha) / 4.

    # shift the and scale the leja points
    leja_x = shift + scale*leja_x

    # evaluate the div. diffs at the shifted leja points
    coeffs_exp = leja_coeffs_exp(leja_x, shift, scale)
    coeffs_ts = dd_exp_taylor(leja_x, shift, scale)
    assert _conv_allclose(coeffs_exp, coeffs_ts, rtol=1e-6)


def test_dd_exp_mccurdy():
    """
    tests from https://doi.org/10.2307/2008291
    """
    # table 2.4.4.1
    leja_x = np.array([-16.0, -12., -8., -4., 0., 4., 8., 12., 16.])
    coeffs_exp = leja_coeffs_exp(leja_x, 0.0, 1.0)
    coeffs_ts = dd_exp_taylor(leja_x, 0.0, 1.0)
    expected = np.asarray([
        1.125351e-7,
        0.150792e-5,
        0.101027e-4,
        0.451239e-4,
        0.151160e-3,
        0.405094e-3,
        0.904679e-3,
        0.173175e-2,
        0.290059e-2])
    assert _conv_allclose(expected, coeffs_ts, rtol=1e-6)
    assert _conv_allclose(expected, coeffs_exp, rtol=1e-6)

    # table 2.4.2.2
    leja_x = np.arange(-24, 24+3, 3) * 1j
    coeffs_exp = leja_coeffs_exp(leja_x, 0.0, 1.0)
    coeffs_ts = dd_exp_taylor(leja_x, 0.0, 1.0)
    expected_dd_16 = 0.699024e-16 + 0.0j
    expected_dd_2 = -0.121109 - 0.184993j
    expected_dd_1 = -0.580745 + 0.323969j
    assert np.isclose(coeffs_ts[16], expected_dd_16, atol=1e-5)
    assert np.isclose(coeffs_ts[1], expected_dd_1, atol=1e-5)
    assert np.isclose(coeffs_ts[2], expected_dd_2, atol=1e-5)


def test_leja_reordering():
    """
    test the leja reordering procedure for real and complex inputs
    """
    # generate real leja points
    leja_x = gen_leja_fast(-2.0, 2.0, 4)
    # randomize points
    leja_x_rng = deepcopy(leja_x)
    np.random.shuffle(leja_x_rng)
    # reorder back to a leja sequence
    leja_x_ord = leja_seq_reordering(leja_x_rng)
    assert np.allclose(leja_x_ord, leja_x)
    leja_c = leja_x * (0.0 + 1.0j)
    leja_c_rng = deepcopy(leja_c)
    np.random.shuffle(leja_c_rng)
    leja_c_ord = leja_seq_reordering(leja_c_rng)
    assert np.allclose(leja_c_ord, leja_c)


def test_gen_leja_conjugate():
    """
    Ensure leja points on the ellipse come in conjugate pairs
    """
    for leja_c in [0.1, 1.0, 2.0, 8.0]:
        for n in [4, 6, 10, 40, 80]:
            zt = gen_leja_conjugate(n, -2.0, 2.0, leja_c)[0]
            assert np.allclose(zt[0:2].imag, 0.0j)
            # ensure conjugate pairs after first 2 (real) points
            assert np.allclose(zt[2:n:2][0:n-1].imag, -zt[3:n:2][0:n-1].imag)
            assert np.allclose(zt[2:n:2][0:n-1].real, zt[3:n:2][0:n-1].real)
