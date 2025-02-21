"""
Implements the phi-functions
"""
from functools import partial
import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx

import warnings

def f_phi_k(z: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z) for dense Z
    """
    mach_eps = jnp.finfo(z.dtype).eps
    phi_k, err_est = f_phi_k_inv(z, k=k, eps=mach_eps)

    if err_est > np.sqrt(mach_eps):
        warnings.warn("the argument of phi_k is close to singular, and f_phi_k is inaccurate.\n" \
                      f"estimated error={err_est}. use f_phi_k_ext instead.")

    return phi_k

def _validate_args(z: jax.Array, k: int):
    assert k >= 0
    N, M = z.shape
    assert N == M
    return N

@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_inv(z: jax.Array, k: int, eps: float) -> (jax.Array, float):
    """
    Computes phi_k(Z) for dense Z, using a formula involving the inverse of Z.
    Returns an error estimate to warn about nearly singular Z.
    """
    N = _validate_args(z, k)

    # phi_0 = exp
    phi_k = jax.scipy.linalg.expm(z)
    err_est = eps
    if k == 0:
        return phi_k, err_est
    else:
        # compute a qr decomposition instead of inverse
        Qz, Rz = jax.scipy.linalg.qr(z)
        r_min = jnp.min(jnp.abs(jnp.diag(Rz)))
        I = jnp.eye(N)
        #DEBUG
        #jax.debug.print("Qz: {Q}, Rz: {R}, r_min: {r_min}", Q=Qz, R=Rz, r_min=r_min)
        #jax.debug.print("phi_0: {M}", M=phi_k)
        for ki in range(1, k+1):
            kfact = 1./jax.scipy.special.factorial(ki-1)
            phi_k = jax.scipy.linalg.solve_triangular(Rz, Qz.T @ (phi_k - kfact*I))
            err_est = err_est / r_min

            #jax.debug.print("phi_k: {M}, err_est: {err_est}", M=phi_k, err_est=err_est)
    return phi_k, err_est

@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_ext(z: jax.Array, k: int, return_all: bool=False) -> jax.Array:
    """
    Computes phi_k(Z) for dense Z, using the stable but more expensive extension formula
    """
    N = _validate_args(z, k)

    if k > 0:
        z_ext_k = jnp.block([[z], [jnp.zeros(((k-1)*N, N))]])
        z_ext = jnp.block([[z_ext_k, jnp.eye(k*N)], [jnp.zeros((N, (k+1)*N))]])
    else:
        z_ext = z

    phi_ks = jax.scipy.linalg.expm(z_ext)

    if return_all:
        phi_k = phi_ks[:N,:].reshape((N,k+1,N))
        phi_k = phi_k.swapaxes(0, 1) # swap the k index to first axes
        return phi_k
    else:
        phi_k = phi_ks[:N,-N:]
        return phi_k

@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_poly_all(z: jax.Array, k: int, poly_deg: int=4) -> list[jax.Array]:
    """
    Computes phi_k(Z) for dense Z, using a Taylor polynomial
    """
    N = _validate_args(z, k)
    assert(poly_deg >= k+1)

    fact_k = jax.scipy.special.factorial(k)
    zpow_kfac = z / fact_k / (k+1.)
    poly_approx = jnp.eye(N)/fact_k + zpow_kfac

    for j in range(k+2, poly_deg+1):
        zpow_kfac = z @ zpow_kfac / j
        poly_approx = poly_approx + zpow_kfac

    phi_ks = [None] * (k+1)
    phi_ks[k] = poly_approx
    fact_j = fact_k / k
    for j in range(k-1, -1, -1):
        # recursion formula phi_j(z) = 1/j! + z phi_{j+1}(z)
        phi_ks[j] = jnp.eye(N)/fact_j + z @ phi_ks[j+1]
        fact_j = fact_j / j

    return phi_ks

@partial(jax.jit, static_argnums=(1,))
def f_phi_k_sq_all(z: jax.Array, k: int) -> list[jax.Array]:
    """
    Computes phi_k(Z) for dense Z, using the scaling and squaring relations
    """
    N = _validate_args(z, k)

    # use infty matrix norm instead of spectral radius to determine scaling
    theta = jnp.linalg.norm(z, ord=np.inf)
    # TODO: determine the optimal initial polynomial degree and the number of squarings
    scale_fact = 16
    init_poly_deg = 4
    Nscale = jnp.floor(jnp.log2(theta * scale_fact)).astype(int)
    tt_N = 2 ** Nscale

    # compute the initial approximation of the phi functions for scaled z
    phi_ks = f_phi_k_poly_all(z / tt_N, k, poly_deg=init_poly_deg)

    # determine scaling constants
    zero_to_k = jax.lax.iota(z.dtype, k+1)
    scalings = 1. / (jax.scipy.special.factorial(zero_to_k[:,None] - zero_to_k[None,:]) * 2. ** zero_to_k[:,None])

    #jax.debug.print("scalings={a}, Nscale={b}", a=scalings, b=Nscale)

    def sq_step(counter, phi_ks):
        for j in range(k, 0, -1):
            # first term in the sum and last correction
            phi_ks[j] = (phi_ks[0] @ phi_ks[j] + phi_ks[j]) * scalings[j,j]
            for jj in range(j-1, 0, -1):
                # remaining corrections
                phi_ks[j] = phi_ks[j] + phi_ks[jj] * scalings[j,jj]
        # traditional squaring of exp
        phi_ks[0] = phi_ks[0] @ phi_ks[0]
        return phi_ks

    phi_ks = jax.lax.fori_loop(0, Nscale, sq_step, phi_ks)

    return phi_ks

def f_phi_k_sq(z: jax.Array, k: int, return_all: bool=False) -> jax.Array:

    phi_ks = f_phi_k_sq_all(z, k)
    if return_all:
        return jnp.array(phi_ks)
    else:
        return phi_ks[k]

## methods for phi_k(A)B

def _validate_args_appl(z: jax.Array, b: jax.Array, k: int):
    assert k >= 0
    assert len(z.shape) == 2
    N, N1 = z.shape
    assert N == N1
    if len(b.shape) == 1:
        N2 = b.shape[0]
        M = 1
        B = b[:,None]
        assert N2 == N
    else:
        assert len(b.shape) == 2
        N2, M = b.shape
        B = b
        assert N2 == N

    return N, M, B

@partial(jax.jit, static_argnums=(2,))
def f_phi_k_appl(z: jax.Array, b: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z)B for dense Z and dense B, using an extension formula
    """
    N, M, B = _validate_args_appl(z, b, k)

    if k > 0:
        z_ext = jnp.block([ [z, B, jnp.zeros((N, (k-1)*M))],
                            [jnp.zeros(((k-1)*M, N+M)), jnp.eye((k-1)*M)],
                            [jnp.zeros((M, N+k*M))] ])
        phi_k_ext = f_phi_k_sq(z_ext, k=0)
        phi_kb = phi_k_ext[:N,-M:].reshape(b.shape)
    else:
        phi_kb = jax.scipy.linalg.expm(z) @ b

    return phi_kb
