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
    print(f"jit-compiling MGS")

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
        qs = qs.at[:, k+1].set(qv / norm_v)
        return qs

    not_breakdown = (norm_v > breakdown_tol)

    update_last = not_final_it & not_breakdown
    qs = lax.cond(update_last,
        lambda k, qv, qs, norm_v: body_subdiag(k, qv, qs, norm_v),
        lambda k, qv, qs, norm_v: qs,
        k, qv, qs, norm_v)

    return hs, qs, not_breakdown

@partial(jax.jit, static_argnums=(3,))
def arnoldi_lop_jit(a_lo: LinOp, a_scale: float, b: jax.Array, m: int, iom: int) -> (jax.Array, jax.Array, int):
    """
    Args:
        a_lo: linear operator
        a_scale: scaling factor
        b: right hand side
        m: number of Krylov vectors to produce
        iom: number of (incomplete) orthogonalizations

    Return:
        hs: hessenberg
        qs: orthonormal basis of krylov subspace
        breakdown_k: iteration k where happy breakdown occurred, or m
    """
    print(f"jit-compiling Arnoldi")

    b_nrows = b.shape[0]
    m = min(b_nrows, m)
    hs = jax.numpy.zeros((m, m))
    qs = jax.numpy.zeros((b_nrows, m))

    m = hs.shape[0]
    norm_b = jnp.linalg.norm(b, 2)
    not_breakdown = (norm_b > 0.)
    q0 = lax.cond(not_breakdown, \
                  lambda b, norm_b: b / norm_b, \
                  lambda k, norm_b: b, \
                  b, norm_b)
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

    hs, qs, breakdown_k, _ = lax.while_loop(cond_arnoldi, body_arnoldi, (hs, qs, k, not_breakdown))
    return hs, qs, breakdown_k

def arnoldi_lop(a_lo: LinOp, a_scale: float, b: jax.Array, m: int, iom: int) -> (jax.Array, jax.Array):

    hs, qs, breakdown_k = arnoldi_lop_jit(a_lo, a_scale, b, m, iom)

    #print(jnp.diag(hs, -2), jnp.diag(hs, -1), jnp.diag(hs, 0), jnp.diag(hs, 1))
    #print(f"Arnoldi had breakdown at {breakdown_k}, m={m}, iom={iom}.")

    return qs[:,:breakdown_k], hs[:breakdown_k,:breakdown_k]
