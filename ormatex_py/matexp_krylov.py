"""
Krylov methods to calculate exp(A*dt)*v and
phi_k(A*dt)*v with sparse A
"""
from collections.abc import Callable
import jax
from jax import numpy as jnp
import equinox as eqx

# internal imports
from ormatex_py.arnoldi_jax import arnoldi_lop
from ormatex_py.matexp_phi import f_phi_k_appl


def matexp_linop(a_lo: Callable, dt: float, v0: jax.Array, max_krylov_dim: int, iom: int=2) -> jax.Array:
    """
    Computes exp(A*dt)*v where A is a sparse linop
    """
    return phi_linop(a_lo, dt, v0, k=0, max_krylov_dim=max_krylov_dim, iom=iom)


def phi_linop(a_lo: Callable, dt: float, v0: jax.Array, k: int, max_krylov_dim: int, iom: int=2):
    """
    Computes phi_k(A*dt)*v where A is a sparse linop
    """
    (q, h, _) = arnoldi_lop(a_lo, 1.0, v0, max_krylov_dim, iom)

    unit_vec = jnp.zeros((h.shape[0],))
    unit_vec = unit_vec.at[0].set(1.0)

    phi_k_e1 = f_phi_k_appl(dt*h, unit_vec, k)
    beta = jnp.linalg.norm(v0, 2)

    return beta * (q @ phi_k_e1)
