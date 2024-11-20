"""
Arnoldi impl
TODO: The linear operator, a_lo, needs to be jit-able
"""
from functools import partial
import jax
from jax import lax
from jax import numpy as jnp

from ormatex_py.ode_sys import LinOp

# inner ortho procedure, modifies hs and qs in-place
#@partial(jax.jit, static_argnums=(4,5,6,))
def arnoldi_mgs_lop(a_lo: LinOp, hs: jax.Array, qs: jax.Array,
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
    breakdown_tol = 1e-12
    # iom_depth = max(k - iom, 0)
    iom_depth = jnp.maximum(k - iom, 0)

    # current vector to ortho against
    q_col = qs[:, k]

    # matvec
    qv = a_lo(q_col) * a_scale
    # TODO: re-enable
    # qv = m_lo(qv)

    # incomplete ortho
    #for i in range(iom_depth, k+1):
    #    qci = qs[:, i]
    #    ht = jnp.dot(qv, qci)
    #    hs = hs.at[i, k].set(ht)
    #    # this makes a copy of qv I think?
    #    qv -= qci * ht
    def body_ortho(i, args):
        # unpack args
        hs, qs, qv = args
        qci = qs[:, i]
        ht = jnp.dot(qv, qci)
        hs = hs.at[i, k].set(ht)
        qv -= qci * ht
        return hs, qs, qv
    hs, qs, qv = lax.fori_loop(iom_depth, k+1, body_ortho, (hs, qs, qv))

    norm_v = jnp.linalg.norm(qv, 2)
    # if k+1 < n:
    #     hs = hs.at[k+1, k].set(norm_v)
    test_a = k+1 < n
    hs = lax.cond(test_a,
        lambda k, hs, norm_v: hs.at[k+1, k].set(norm_v), \
        lambda k, hs, norm_v: hs, \
        k, hs, norm_v)

    # if k+1 < n and norm_v > breakdown_tol:
    #     qv *= 1./norm_v
    #     qs = qs.at[:, k+1].set(qv)
    def body_brkdwn(k, qv, qs, norm_v):
        qv *= 1./norm_v
        qs = qs.at[:, k+1].set(qv)
        return (qv, qs)
    test_b = (k+1 < n) & (norm_v > breakdown_tol)
    qv, qs = lax.cond(test_b,
        lambda k, qv, qs, norm_v: body_brkdwn(k, qv, qs, norm_v),
        lambda k, qv, qs, norm_v: (qv, qs),
        k, qv, qs, norm_v)

    return hs, qs, (norm_v <= breakdown_tol)

#@partial(jax.jit, static_argnums=(3,4,))
def arnoldi_lop(a_lo: LinOp, a_scale: float, b: jax.Array, m: int, iom: int) -> (jax.Array, jax.Array, int):
    b_nrows = b.shape[0]
    m = min(b_nrows, m)
    hs = jax.numpy.zeros((m,m))
    qs = jax.numpy.zeros((b_nrows, m))
    q0 = b / jnp.linalg.norm(b, 2)
    qs = qs.at[:, 0].set(q0.flatten())

    breakdown_n = 0
    for k in range(0, m):
        hs, qs, breakdown_flag = arnoldi_mgs_lop(
                a_lo, hs, qs, a_scale, k, m, iom)
        breakdown_n += 1
        if breakdown_flag:
            break

    return (qs[0:b_nrows, 0:breakdown_n], hs[0:breakdown_n, 0:breakdown_n], breakdown_n)
