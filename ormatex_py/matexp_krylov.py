"""
Krylov methods to calculate exp(A*dt)*v and
phi_k(A*dt)*v with sparse A
"""
from scipy.sparse.linalg import LinearOperator
from scipy.special import factorial
from typing import Callable, List
import jax
import numpy as np
from jax import numpy as jnp
import equinox as eqx
# internal imports
from ormatex_py.arnoldi_jax import arnoldi_lop
from ormatex_py.matexp_phi import f_phi_k, f_phi_k_ext


def matexp_linop(a_lo: Callable, dt: float, v0: jax.Array, max_krylov_dim: int, iom: int=2):
    """
    Computes exp(A*dt)*v where A is a sparse linop
    """
    (q, h, _) = arnoldi_lop(a_lo, dt, v0, max_krylov_dim, iom)
    matexp = jax.scipy.linalg.expm(h)
    beta = jnp.linalg.norm(v0, 2)
    unit_vec = jnp.zeros((matexp.shape[0],1))
    unit_vec = unit_vec.at[0,0].set(1.0)
    tmp = q @ (matexp @ unit_vec)
    return beta * tmp.flatten()


def phi_linop(a_lo: Callable, dt: float, v0: jax.Array, k: int, max_krylov_dim: int, iom: int=2):
    """
    Computes phi_k(A*dt)*v where A is a sparse linop
    """
    (q, h, _) = arnoldi_lop(a_lo, 1.0, v0, max_krylov_dim, iom)
    phi_k = f_phi_k_ext(dt*h, k)
    beta = jnp.linalg.norm(v0, 2)
    unit_vec = jnp.zeros((phi_k.shape[0],1))
    unit_vec = unit_vec.at[0,0].set(1.0)
    tmp = q @ (phi_k @ unit_vec)
    return beta * tmp.flatten()


def phipm_unstable(a_lo: Callable, dt: float, vb: List[jax.Array], p: int, max_krylov_dim: int, iom: int=2, n_steps: int=1):
    """
    Computes linear combinations of phi functions, using the
    phipm approach. Ref:

    J. Niesen, W.M. Wright, Algorithm 919:
        a Krylov subspace algorithm for evaluating the
        phi-functions appearing in exponential integrators,
        ACM Trans. Math. Softw. 38 (3) (2012) 22.

    Computes:
    \tau*phi_0(A*\tau)*v_0 +
    \tau^1*phi_1(A*\tau)*v_1 +
    \tau^2*phi_2(A*\tau)*v_2
    where A is a sparse linop

    Args:
        a_lo: linear operator.  implements __call__ for matvec
        vb: list of rhs vectors [v_0, ... v_p] where p is the highest order phi fn to compute.
        dt: step size.  TODO: support list of tau
        p: int. highest order phi fn to compute.
    """
    # TODO: this substepping needs to be replaced
    # step sizes
    tau = np.ones(n_steps)
    tau = tau / np.sum(tau)

    n = vb[0].shape[0]
    w_phi = vb[0]

    # substepping procedure
    for i, tau_i in enumerate(tau):
        w = np.zeros((n,p+1))
        w[:, 0] = w_phi
        for j in range(1, p+1):
            for l in range(0, p-j+1):
                w[:, j] += (np.power(tau_i, l) / factorial(l)) * vb[j+l]
            w[:, j] += dt * a_lo(w[:, j-1])
        w_sum = np.zeros(n)
        for j in range(0, p-1+1):
            w_sum += (np.power(tau_i, j) / factorial(j)) * w[:, j]

        w_phi_update = np.power(tau_i, p) * \
                phi_linop(a_lo, tau_i * dt, w[:, p], p,
                          max_krylov_dim=max_krylov_dim, iom=iom)
        w_phi = w_phi_update + jnp.asarray(w_sum)

    return w_phi
