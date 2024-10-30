"""
Krylov methods to calculate exp(A*dt)*v and
phi_k(A*dt)*v with sparse A
"""
from collections.abc import Callable
import jax
import numpy as np
from jax import numpy as jnp
import equinox as eqx

# internal imports
from ormatex_py.ode_sys import AugMatrixLinop
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


def kiops_fixedsteps(a_lo: Callable, dt: float, vb: list[jax.Array], max_krylov_dim: int, iom: int=2, n_steps: int=1):
    r"""
    Method based roughly on simplified KIOPS with fixes stepsize
    and not Krylov adaptivity.  TODO: add adaptivity routines.
    This avoids the substepping procedure in phipm by computing
    the phi-vection products as:

    .. math::

        w(\tau) = \sum_{j=0}^p \tau^j \varphi_j(\tau A) b_j

        w(\tau) = \exp(\tau \tilde A)v

    with :math:` v = [b_0, e_1]^T `
    and
    .. math::

        \tilde A = [[A, B],[0, K]]

    where A=a_lo is NxN,
    B = vb[::-1] is Nxp,
    K = [[0, I_{p-1}],[0, 0]] is pxp,
    :math:` \tilde A ` is N+p x N+p
    """
    p = len(vb) - 1
    # fixed stepsize
    tau_i = 1.0
    n = vb[0].shape[0]

    # build B
    b = jnp.vstack(vb[::-1][:-1]).T  # [::-1] reverse view

    # build \tilde A
    k = np.zeros((p,p))
    k[0:p-1, 1:] = np.eye(p-1)
    k = jnp.asarray(k)
    a_tilde_lo = AugMatrixLinop(a_lo, dt, b, k)

    unit_vec = np.zeros(p)
    unit_vec[-1] = 1.0
    v = jnp.concat((vb[0], jnp.asarray(unit_vec)))

    w = matexp_linop(a_tilde_lo, tau_i * 1.0, v,
              max_krylov_dim=max_krylov_dim, iom=iom)
    return w[0:n]
