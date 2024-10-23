"""
Krylov methods to calculate exp(A*dt)*v and
phi_k(A*dt)*v with sparse A
"""
from scipy.sparse.linalg import LinearOperator
from typing import Callable, List
import jax
from jax import numpy as jnp
import equinox as eqx
# internal imports
from ormatex_py.arnoldi_jax import arnoldi_lop
from ormatex_py.matexp_phi import f_phi_k


def matexp_linop(a_lo: Callable, dt: float, v0: jax.Array, max_krylov_dim: int, iom: int=2):
    """
    Computes exp(A*dt)*v where A is a sparse linop
    """
    (q, h, _) = arnoldi_lop(a_lo, dt, v0, max_krylov_dim, iom)
    matexp = jax.scipy.linalg.expm(h)
    beta = jnp.linalg.norm(v0, 2)
    unit_vec = jnp.zeros((matexp.shape[0],))
    unit_vec.at[0,0].set(1.0)
    return beta * q * matexp * unit_vec


def phi_linop(a_lo: Callable, dt: float, v0: jax.Array, k: int, max_krylov_dim: int, iom: int=2):
    """
    Computes phi_k(A*dt)*v where A is a sparse linop
    """
    (q, h, _) = arnoldi_lop(a_lo, 1.0, v0, max_krylov_dim, iom)
    phi_k = f_phi_k(dt*h, k)
    beta = jnp.linalg.norm(v0, 2)
    unit_vec = jnp.zeros((phi_k.shape[0],))
    unit_vec.at[0,0].set(1.0)
    return beta * q * phi_k * unit_vec
