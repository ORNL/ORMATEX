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


@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_inv(z: jax.Array, k: int, eps: float) -> (jax.Array, float):
    """
    Computes phi_k(Z) for dense Z, using a formula involving the inverse of Z.
    Returns an error estimate to warn about nearly singular Z.
    """
    assert k >= 0
    N, M = z.shape
    assert N == M
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
    assert k >= 0
    N, M = z.shape
    assert N == M

    if k > 0:
        z_ext_k = jnp.block([[z], [jnp.zeros(((k-1)*N, N))]])
        z_ext = jnp.block([[z_ext_k, jnp.eye(k*N)], [jnp.zeros((N, (k+1)*N))]])
    else:
        z_ext = z

    phi_ks = jax.scipy.linalg.expm(z_ext)

    if return_all:
        phi_k = phi_ks[:N,:]
    else:
        phi_k = phi_ks[:N,-N:]
    return phi_k


@partial(jax.jit, static_argnums=(2,))
def f_phi_k_appl(z: jax.Array, b: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z)B for dense Z and dense B, using an extension formula
    """
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

    if k > 0:
        z_ext = jnp.block([ [z, B, jnp.zeros((N, (k-1)*M))],
                            [jnp.zeros(((k-1)*M, N+M)), jnp.eye((k-1)*M)],
                            [jnp.zeros((M, N+k*M))] ])
        phi_k_ext = jax.scipy.linalg.expm(z_ext)
        phi_kb = phi_k_ext[:N,-M:].reshape(b.shape)
    else:
        phi_kb = jax.scipy.linalg.expm(z) @ b

    return phi_kb
