##############################################################################
# Copyright© 2025 UT-Battelle, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""
This submodule holds polynomial approximation tools for
the matrix exponential and the action of the matrix exponential on
a vector.  A selection of divided difference routines are provided
to compute the polynomial coefficients of the matrix exponential
and related phi-functions.
"""
from functools import partial
import numpy as np
import scipy as sp
import math
import jax
from jax import lax
from jax import numpy as jnp


@jax.jit
def newton_poly_div_diff(x: jax.Array, y: jax.Array):
    """
    Divided difference formula to compute coeffs of the newton polynomial
    given pairs of (x_i, y_i)

    Args:
        x: data support points, where the data is sampled.
        y: value of the function at x points.
    """
    m = len(x)
    a = y.at[:].get() # TODO why not a = y?
    for k in range(1, m):
        a = a.at[k:m].set( (a[k:m] - a[k-1]) / (x[k:m] - x[k-1]) )
    return a


@jax.jit
def leja_coeffs_exp_dd(leja_x: jax.Array, shift: float, scale: float, h: float=1.0):
    """
    Dividied difference computation of the leja polynomial coefficients.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
        h: step fraction
    """
    coeffs = newton_poly_div_diff(leja_x, jnp.exp(h*(shift + scale*leja_x)))
    return coeffs


@jax.jit
def leja_coeffs_exp(leja_x: jax.Array, shift: float, scale: float, h: float=1.0):
    """
    Alternate method to compute the leja polynomial coefficients.
    Reduced roundoff error compared to divided difference formula.
    Ref:

        M. Calari.  Accurate evaluation of divided differences
        for polynomial interpolation of exponential propagators.
        Computing. 80. 2007.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
        h: step fraction
    """
    n_leja = len(leja_x)
    Xi = jnp.diag(leja_x, 0) + jnp.diag(jnp.ones(n_leja-1), -1)
    Xi_shift = jnp.identity(n_leja)*shift + scale*Xi
    # extract the first column
    coeffs = jax.scipy.linalg.expm(h*Xi_shift)[:, 0]
    return coeffs


@partial(jax.jit, static_argnums=(3,))
def expm_taylor(A: jax.Array, shift: float, scale: float, p: int=20):
    r"""
    Taylor expansion of the shifted matrix exp function for
    dense matricies A.

    .. math:

        exp(shift + scale * A) = exp(shift) * exp(scale*A)

    Where A is a matrix.
    We approximate the second term with the TS:

    .. math:

        exp(A) \approx (I + A + A*A/2! + A*A*A/3! ...)

    Args:
        A: square matrix
        shift: argument shift
        scale: argument scale
        p: taylor series polynomial order
    """
    M = scale * A
    ts_expm = jnp.identity(A.shape[0])
    for i in range(p):
        # powers of A
        ts_expm = ts_expm + M / jax.scipy.special.factorial(i + 1)
        M = A @ M
    return jnp.exp(shift) * ts_expm


@jax.jit
def dd_exp_taylor(leja_x, shift, scale, h=1.0, p=20, mu_scale=1.0):
    """
    Divided differences of the shifted and scaled exponenial function
    using a taylor series approximation and scaling and squaring.
    Similar to leja_coeffs_exp but with custom TS impl.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
        h: time step fraction
        p: taylor series polynomial order
    """
    n_leja = len(leja_x)
    Xi = jnp.diag(leja_x, 0) + jnp.diag(jnp.ones(n_leja-1), -1)
    Xi_shift = jnp.identity(n_leja)*shift + scale*Xi

    # compute shift
    mu = np.mean(h*(shift + leja_x * scale)) * mu_scale
    z = Xi_shift - jnp.identity(n_leja)*mu

    # scaling (number of squarings is in powers of 2)
    s_scale = jnp.max(jnp.abs(z))
    # s = jnp.max(jnp.asarray([jnp.ceil(jnp.log2(jnp.linalg.norm(z, 1))-2.42).astype(int),1]))
    s = jnp.max(jnp.asarray([jnp.ceil(jnp.log(s_scale)/jnp.log(2.0)).astype(int),1]))
    hs = 1.0/(2.0**s)

    # compute divided differnces by taylor series approx to expm
    # of the shifted and scaled matrix
    F = expm_taylor(hs*h*z, 0.0, 1.0, p=p)

    def body_f(i, F):
        return F @ F

    # squaring
    # for _ in range(s):
    #    F = F @ F
    F = lax.fori_loop(0, s, body_fun=body_f, init_val=F)

    # shift back and extract the first column
    return jnp.exp(h*mu) * F[:, 0]


@partial(jax.jit, static_argnums=(1,))
def leja_seq_pad_zeros(x, p=2):
    """
    Pads a leja sequence with leading (near) zeros.
    When used with the dd_exp_taylor method,
    The divided differences of the repeated zeros
    are the coefficients of a newton polynomial with
    terms that are approx. the taylor series.

    Args:
        x: leja sequence
        l: number of (near) zeros to pad
    """
    zeros = jnp.linspace(1e-12, 1e-12*p, p)
    return jnp.concat((zeros, x))


def leja_seq_reordering(x):
    """
    Reorders the points in sequence x into a leja sequence.

    .. math:

        z_{M+1} = argmax_{z\\in K} \prod_i^M | z - z_i |

    Args:
        x: an unordered sequence of points in the complex plane
    """
    xi = list(x)
    z = [xi.pop(np.argmax(x)), ]
    while len(xi) > 0:
        # for each canidate point, compute
        # distance to all points in the leja sequnce.
        # dp = []
        # for p in xi:
        #     detp = np.prod(np.abs(p - np.asarray(z)))
        #     dp.append(detp)
        # vectorized version of above
        zr = np.tile(np.asarray(z), (len(xi), 1)).T
        dpv = np.prod(np.abs(zr - np.asarray(xi)), axis=0)
        # the index of the next point in the sequence
        ni = np.argmax(dpv)
        # move canidate point into the leja sequence
        z.append(xi.pop(ni))
    return np.asarray(z)


#####################################################################
# EXPERIMENTAL DIVIDED DIFFERENCE METHODS
#####################################################################
def leja_coeffs_ts_ss(leja_x: jax.Array, shift: float, scale: float, h_in: float=1.0, l: int=0):
    """
    Divided differences of Xi, phi_l(Xi) by taylor series
    and scaling and squaring.
    """
    n = len(leja_x)
    # Scaling factor
    s = max(int( np.ceil( np.log(scale) / np.log(2) ) ), 1)
    h = 1/(2**s)
    # shifted and scaled leja points
    nu = np.mean(shift + scale * leja_x)
    z_o = shift + scale * leja_x
    z = h * (shift + scale * leja_x - nu)

    # initilize F
    F = 0.0 * np.tile(leja_x, (n, 1))
    for i in range(n):
        for j in range(0, i+1):
            F[i, j] = (1.0) / math.factorial(i + l - j)
            F[j, i] = (1.0) / math.factorial(i + l - j)

    for k in range(1, 20):
        for j in range(n-1):
            F[j, j] = z[j] * (F[j, j] / (k + l))
            for i in range(j+1, n):
                F[i, j] = (z[i] * F[i, j] + F[i-1, j]) / \
                        (k + i - j + l)
                F[j, i] = F[i, j] + F[j, i]
    for j in range(0, n):
        F[j, j] = np.exp(z[j])
    # set lower triangle to 0
    F = np.triu(F)

    # Squaring
    R = np.diag([2.0 ** i for i in range(n)])
    for kk in range(1, s+1):
        F = R @ (F @ F) @ np.linalg.inv(R)
        z = 2.0 * z
        for j in range(0, n):
            F[j, j] = np.exp(z[j])
    z_out = z + nu
    # Extract first row from full div. diff. table
    F = np.exp(nu) * F
    dd = F[0, :]
    return dd


def leja_coeffs_ts_phi(leja_x: jax.Array, shift: float, scale: float, h: float=1.0, k: int=0, full: bool=False):
    """
    Divided difference of  phi_k(h*(a+b*xi)) evaluated at xi \\in [-2, 2]
    by taylor series approximation.  Uses the TS(II) method from
    Refs:

        A. McCurdy, K. Ng and B. Parlett. Accurate Computation of Divided differences
        of the exponential function. Mathmatics of Computation. v 43. 1984.

        M. Calari.  Accurate evaluation of divided differences for polynomial
        interpolation of exponential propagators. Computing. 80. 2007.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
        h: step fraction
        k: the phi function order
    """
    m = len(leja_x)
    F = 0.0 * np.tile(leja_x, (m, 1))
    z = shift + scale * leja_x
    # initilize F
    for i in range(m):
        for j in range(0, i+1):
            F[i, j] = ((h * scale) ** i) / sp.special.factorial(i + k - j)
            F[j, i] = ((h * scale) ** i) / sp.special.factorial(i + k - j)
    for l in range(2, 17):
        for j in range(m-1):
            F[j, j] = h * (z[j]) * (F[j, j] / (l + k - 1))
            for i in range(j+1, m):
                F[j, i] = h * ((z[i]) * F[j, i] + scale * F[j, i-1]) / (l + i - j + k - 1)
                F[i, j] = F[i, j] + F[j, i]
    # for j in range(0, m):
    #     F[j, j] = np.exp(h*(z[j]))
    np.fill_diagonal(F, np.exp(h*(z)))
    # set upper triangle to 0
    F = np.tril(F)
    if full:
        return F
    # extract the first column
    coeffs = F[:, 0]
    return coeffs


@jax.jit
def leja_coeffs_ts_phi_jax(leja_x: jax.Array, shift: float, scale: float, h: float=1.0, k: int=0):
    """
    JAX implementation of TS(II).
    See reference numpy implementation in :method:`leja_coeffs_ts_phi`

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
        h: time step fraction
        k: the phi function order
    """
    print("jit compiling leja_coeffs_ts")
    m = len(leja_x)
    F = jnp.zeros((m, m), dtype=jnp.complex64)
    z = shift + scale * leja_x
    # initilize F
    for i in range(m):
        for j in range(0, i+1):
            F = F.at[i, j].set( ((h * scale) ** i) / jax.scipy.special.factorial(i + k - j) )
            F = F.at[j, i].set( ((h * scale) ** i) / jax.scipy.special.factorial(i + k - j) )
    for l in range(2, 18):
        for j in range(m-1):
            F = F.at[j, j].set( h * (z[j]) * (F[j, j] / (l + k - 1)) )
            for i in range(j+1, m):
                F = F.at[j, i].set( h * ((z[i]) * F[j, i] + scale * F[j, i-1]) / (l + i - j + k - 1) )
                F = F.at[i, j].set( F[i, j] + F[j, i] )
    # for j in range(0, m):
    #     F = F.at[j, j].set( jnp.exp(h*(z[j])) )
    jnp.fill_diagonal(F,  jnp.exp(h*(z)) )
    # extract the first column
    coeffs = F.at[:, 0].get()
    return coeffs


def leja_coeffs_ts_phi_substep(
        leja_x: jax.Array, shift: float, scale: float, k: int=0):
    """
    Substepping procedure for the divided differences using
    the ts_phi method.
    """
    n_leja = len(leja_x)
    steps = int(np.ceil(scale))
    tau = 1. / scale
    Xi = jnp.diag(leja_x, 0) + jnp.diag(jnp.ones(n_leja-1), -1)
    H_m = jnp.identity(n_leja)*shift + scale*Xi
    e_1 = np.zeros(n_leja)
    e_1[0] = 1.
    # y_{n+1} = exp(\tau A)(y_n)
    y = leja_coeffs_ts_phi(leja_x, shift, scale, tau, k, full=True) @ e_1
    for s in range(steps-1):
        y = leja_coeffs_ts_phi(leja_x, shift, scale, tau, k, full=True) @ y
    return y


def leja_coeffs_dd_phi(leja_x: jax.Array, shift: float, scale: float, h: float=1.0, l: int=0):
    """
    Fast divided difference calculation for phi_l(Z) evaluated at z=(a+b*xi)
    using the dd_phi routine from Zivovich (2019).
    Avoids expm call from leja_coeffs_exp method.

    Ref:
        Fast and accurate computation of divided differences for analytic
        functions, with an application to the exponential function.
        F. Zivcovich. Dolomites Research Notes on Approximation. v 12. 2019.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
        h: time step fraction
        l: the phi function order
    """
    Xi_shift = h * (shift + scale*leja_x)

    z = np.concatenate((np.zeros(l), Xi_shift.flatten()))
    # shift abscissa to be centered around 0
    mu = np.mean(z)
    z = z - mu
    n = len(z) - 1
    N = n + 30

    F = np.zeros((n+1, n+1), dtype=np.complex64)
    for i in range(n):
        F[i+1:n+1, i] = z[i] - z[i+1:n+1]

    # Compute scaling factor
    s = max(int(np.ceil(np.max(np.abs(F)) / 3.5)), 1)

    # Compute divided differences at the scaled points
    dd = np.concatenate(([1], 1.0 / np.cumprod((np.arange(1, N+1)) * s,
                                               dtype=np.complex128)))
    for j in range(n, -1, -1):
        for k in range(N-1, n-j-1, -1):
            dd[k] = dd[k] + z[j] * dd[k+1]
        for k in range(n-j-1, -1, -1):
            dd[k] = dd[k] + F[k+j+1, j] * dd[k+1]
        F[j, j:n+1] = dd[:n-j+1]
    np.fill_diagonal(F, np.exp(z / s))
    F = np.triu(F)
    dd = F[0, :]

    # Squaring
    for k in range(1, s):
        dd = np.dot(dd, F)

    dd =  np.exp(mu) * dd[l:n+1]
    return dd
