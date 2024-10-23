"""
Implements the phi-functions
"""
import jax
from jax import numpy as jnp


@jax.jit
def f_phi_k(z: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z) for dense Z
    """
    assert k >= 0
    # phi_0
    phi_k = jax.scipy.linalg.expm(z)
    if k == 0:
        return phi_k
    else:
        z_inv = jax.scipy.linalg.inv(z)
        id = jnp.eye(z.shape[0])
        for i in range(1, k+1):
            phi_k = z_inv * (phi_k - (1./jax.scipy.special.factorial(i))*id)
    return phi_k
