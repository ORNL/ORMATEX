"""
Arnoldi impl
TODO: The linear operator, a_lo, needs to be jit-able
"""
from scipy.sparse.linalg import LinearOperator
from typing import Callable
from functools import partial
import jax
from jax import numpy as jnp


@partial(jax.jit(static_argnums=(5,6)))
def arnoldi_mgs_lop(a_lo: Callable, hs: jax.Array, qs: jax.Array,
             a_scale: float, k: int, n: int, iom: int) -> bool:
    """
    TODO: The linear operator, a_lo, must be jit-able!

    Args:
        a_lo: linear operator
        hs: hessenberg
        qs: orthonormal basis of krylov subspace

    Return:
        bool. true if happy breakdown
    """
    breakdown_tol = 1e-14
    iom_depth = max(k - iom, 0)

    # current vector to ortho against
    q_col = qs[:, k]

    # matvec
    qv = a_lo(q_col)

    # ortho
    # h = hs[:, k]
    for i in range(iom_depth, k+1):
        qci = qs[:, i]
        ht = jnp.dot(qv, qci)
        hs.at[i, k].set(ht)
        # this makes a copy of qv I think?
        qv -= qci * ht

    norm_v = jnp.linalg.norm(qv, 2)
    if k+1 < n:
        hs.at[k+1, k].set(norm_v)

    if k+1 < n and norm_v > breakdown_tol:
        qv *= 1./norm_v
        qs.at[:, k+1].set(qv)

    if norm_v <= breakdown_tol:
        return True
    return False


@partial(jax.jit(static_argnums=(3,4,)))
def arnoldi_lop(a_lo: Callable, a_scale: float, b: jax.Array, n: int, iom: int) -> (jax.Array, jax.Array, int):
    b_nrows = b.shape[0]
    hs = jax.numpy.zeros(n, n)
    qs = jax.numpy.zeros(b_nrows, n)
    q0 = b / jnp.linalg.norm(b, 2)
    breakdown_n = 0
    for k in range(0, n):
        breakdown_flag = arnoldi_mgs_lop(a_lo, hs, qs, a_scale, k, n, iom)
        breakdown_n += 1
        if breakdown_flag:
            break

    return (qs[0:b_nrows, 0:breakdown_n], hs[0:breakdown_n, 0:breakdown_n], breakdown_n)
