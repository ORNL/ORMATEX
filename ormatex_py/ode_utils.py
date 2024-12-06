"""
Helper functions for ODE integrators.
"""
import jax


def stack_u(u: jax.Array, n: int):
    return u.reshape((-1, n), order='F')

def flatten_u(u: jax.Array):
    # use column major ordering
    return u.reshape((-1, ), order='F')

