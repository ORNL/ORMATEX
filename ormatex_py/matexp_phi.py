"""
Implements the phi-functions
"""
from functools import partial
import jax
from jax import numpy as jnp


@partial(jax.jit, static_argnums=(1,))
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
        z_inv = jnp.linalg.inv(z)
        id = jnp.eye(z.shape[0])
        #DEBUG
        #print("z_inv: ", z_inv)
        #print("phi_0: ", phi_k)
        for i in range(1, k+1):
            ifact = 1./jax.scipy.special.factorial(i-1)
            phi_k = z_inv @ (phi_k - ifact*id)
            #print("phi_k: ", phi_k)
    return phi_k
