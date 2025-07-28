"""
Krylov methods to calculate exp(A*dt)*v and
phi_k(A*dt)*v with sparse A
"""
import jax
import numpy as np
from jax import numpy as jnp
import equinox as eqx

# internal imports
from ormatex_py.ode_sys import LinOp, AugMatrixLinOp
from ormatex_py.arnoldi_jax import arnoldi_lop
from ormatex_py.matexp_phi import f_phi_k_appl, f_phi_k


def matexp_linop(a_lo: LinOp, dt: float, v0: jax.Array, max_krylov_dim: int, iom: int=100) -> jax.Array:
    """
    Computes exp(A*dt)*v where A is a sparse linop
    """
    return phi_linop(a_lo, dt, v0, k=0, max_krylov_dim=max_krylov_dim, iom=iom)


def phi_linop(a_lo: LinOp, dt: float, v0: jax.Array, k: int, max_krylov_dim: int, iom: int=100) -> jax.Array:
    """
    Computes phi_k(A*dt)*v where A is a sparse linop
    """

    norm_v0 = jnp.linalg.norm(v0, 2)
    if norm_v0 > 0:
        v = v0 / norm_v0
        (q, h) = arnoldi_lop(a_lo, dt, v, max_krylov_dim, iom)
        assert(h.shape[0] > 0)

        unit_vec = jnp.zeros((h.shape[0],))
        unit_vec = unit_vec.at[0].set(1.0)

        phi_k_e1 = f_phi_k_appl(h, unit_vec, k)

        phi_k_v0 = norm_v0 * (q @ phi_k_e1)
    else:
        # v0 is zero, thus phi_k_v as well
        phi_k_v0 = jnp.zeros_like(v0)

    #TODO: need a better logic for handling the tolerances/size of krylov_space

    return phi_k_v0

def kiops_fixedsteps(a_lo: LinOp, dt: float, vb: list[jax.Array], max_krylov_dim: int, iom: int=100, n_steps: int=1) -> jax.Array:
    r"""
    Method based roughly on simplified KIOPS with fixes stepsize
    and not Krylov adaptivity.  TODO: add adaptivity routines.
    This avoids the substepping procedure in phipm by computing
    the phi-vector products as:

    .. math::

        w(\tau) = \sum_{j=0}^p \tau^j \varphi_j(\tau A) b_j

        w(\tau) = \exp(\tau \tilde A)v

    with :math:` v = [b_0, e_1]^T `
    and
    .. math::

        \tilde A = [[A, B],[0, K]]

    where A = a_lo is NxN,
    B = vb[:0:-1] is Nxp,
    K = [[0, I_{p-1}],[0, 0]] is pxp,
    :math:` \tilde A ` is N+p x N+p
    """
    p = len(vb) - 1
    # fixed stepsize
    tau_i = 1.0
    n = vb[0].shape[0]

    norms_vb = [jnp.linalg.norm(v, 2) for v in vb]
    norm_vb = max(norms_vb)

    if norm_vb > 0:
        # build B
        # [:0:-1] reverse view from -1 down to 1
        # normalize by the largest vector in vb
        b = jnp.vstack(vb[:0:-1]).T / norm_vb

        # build \tilde A
        k = np.zeros((p,p))
        k[0:p-1, 1:] = np.eye(p-1)
        k = jnp.asarray(k)
        a_tilde_lo = AugMatrixLinOp(a_lo, dt, b, k)

        unit_vec = np.zeros(p)
        unit_vec[-1] = 1.0
        v = jnp.concat((vb[0] / norm_vb, jnp.asarray(unit_vec)))

        w = matexp_linop(a_tilde_lo, tau_i, v,
                         max_krylov_dim=max_krylov_dim, iom=iom)

        sum_phi_ks = norm_vb * w[0:n]
    else:
        # vb is zero, thus sum_phi_ks as well
        sum_phi_ks = jnp.zeros_like(vb[0])

    return sum_phi_ks
