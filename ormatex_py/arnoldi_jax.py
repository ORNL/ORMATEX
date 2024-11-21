"""
Arnoldi impl
TODO: The linear operator, a_lo, needs to be jit-able
"""
from functools import partial
import jax
from jax import lax
from jax import numpy as jnp

from ormatex_py.ode_sys import LinOp

# inner ortho procedure, based on MGS
@jax.jit
def arnoldi_mgs_lop(a_lo: LinOp, a_scale: float, hs: jax.Array, qs: jax.Array, k: int, iom: int) -> bool:
    """
    Args:
        a_lo: linear operator
        a_scale: scaling factor
        hs: hessenberg
        qs: orthonormal basis of krylov subspace
        k:  step of Arnoldi

    Return:
        hs: modified hessenberg
        qs: modified ONB
        bool. true if no happy breakdown
    """
    print(f"jit-compiled MGS")

    m = hs.shape[0]
    breakdown_tol = 1e-12
    iom_depth = jnp.maximum(k - iom, 0)

    # current vector to ortho against
    q_k = qs[:, k]

    # matvec: qv = a_scale * A @ q_k
    qv = a_lo(q_k) * a_scale

    # TODO: re-enable Mass matrix
    # qv = m_lo(qv)

    # incomplete orthogonalization (MGS)
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

    not_final_it = k+1 < m
    hs = lax.cond(not_final_it, \
                  lambda k, hs, norm_v: hs.at[k+1, k].set(norm_v), \
                  lambda k, hs, norm_v: hs, \
                  k, hs, norm_v)

    def body_subdiag(k, qv, qs, norm_v):
        qv *= 1./norm_v
        qs = qs.at[:, k+1].set(qv)
        return (qv, qs)

    not_breakdown = (norm_v > breakdown_tol)

    update_last = not_final_it & not_breakdown
    qv, qs = lax.cond(update_last,
        lambda k, qv, qs, norm_v: body_subdiag(k, qv, qs, norm_v),
        lambda k, qv, qs, norm_v: (qv, qs),
        k, qv, qs, norm_v)

    return hs, qs, not_breakdown

@jax.jit
def arnoldi_lop_jit(a_lo: LinOp, a_scale: float, b: jax.Array, hs: jax.Array, qs: jax.Array, iom: int) -> (jax.Array, jax.Array, int):
    """
    Args:
        a_lo: linear operator
        a_scale: scaling factor
        b: right hand side
        hs: hessenberg
        qs: orthonormal basis of krylov subspace
        iom: number of (incomplete) orthogonalizations

    Return:
        hs: modified hessenberg
        qs: modified ONB
        breakdown_k: iteration k where happy breakdown occurred, or hs.shape[0]
    """
    print(f"jit-compiled Arnoldi")

    m = hs.shape[0]
    q0 = b / jnp.linalg.norm(b, 2)
    qs = qs.at[:, 0].set(q0.flatten())
    k = 0

    def body_arnoldi(args):
        hs, qs, k, _ = args
        hs, qs, not_breakdown = arnoldi_mgs_lop(
            a_lo, a_scale, hs, qs, k, iom)
        k += 1
        return hs, qs, k, not_breakdown
    def cond_arnoldi(args):
        _, _, k, not_breakdown = args
        return (k < m) & not_breakdown

    hs, qs, breakdown_k, _ = lax.while_loop(cond_arnoldi, body_arnoldi, (hs, qs, k, True))
    return hs, qs, breakdown_k

def arnoldi_lop(a_lo: LinOp, a_scale: float, b: jax.Array, m: int, iom: int) -> (jax.Array, jax.Array, int):
    b_nrows = b.shape[0]
    m = min(b_nrows, m)
    hs = jax.numpy.zeros((m,m))
    qs = jax.numpy.zeros((b_nrows, m))

    hs, qs, breakdown_k = arnoldi_lop_jit(a_lo, a_scale, b, hs, qs, iom)
    #breakdown_k = 0
    #for k in range(0, m):
    #    hs, qs, not_breakdown = arnoldi_mgs_lop(
    #            a_lo, a_scale, hs, qs, k, iom)
    #    breakdown_k = k
    #    if not not_breakdown:
    #        break

    #print(jnp.diag(hs, 1))
    #print(f"Arnoldi had breakdown at {breakdown_k}, m={m}, iom={iom}.")

    return qs[:,:breakdown_k], hs[:breakdown_k,:breakdown_k], breakdown_k
