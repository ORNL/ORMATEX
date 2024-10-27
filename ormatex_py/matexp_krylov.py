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
from ormatex_py.matexp_phi import f_phi_k


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
    phi_k = f_phi_k(dt*h, k)
    beta = jnp.linalg.norm(v0, 2)
    unit_vec = jnp.zeros((phi_k.shape[0],1))
    unit_vec = unit_vec.at[0,0].set(1.0)
    tmp = q @ (phi_k @ unit_vec)
    return beta * tmp.flatten()


def phipm_unstable(a_lo: Callable, dt: float, vb: List[jax.Array], p: int, max_krylov_dim: int, iom: int=2):
    """
    Computes linear combinations of phi functions, using the
    phipm approach. Ref:

    J. Niesen, W.M. Wright, Algorithm 919:
        a Krylov subspace algorithm for evaluating the
        phi-functions appearing in exponential integrators,
        ACM Trans. Math. Softw. 38 (3) (2012) 22.

    NOTE: this is probably not great for high order phi-functions
    due to repeated multiplication of a_lo. Indended for use with
    low order (EPI3) exponential rosenbrock methods only.
    TODO: add adaptive krylov dimension and adaptive substepping.
    TODO: upgrade to KIOPS.

    Computes:
    \tau*phi_0(A*\tau)*v_0 +
    \tau^1*phi_1(A*\tau)*v_1
    \tau^2*phi_2(A*\tau)*v_2
    where A is a sparse linop

    This is routine can save multiple calls to arnoldi by
    evaulating only the highest-order phi-function-vector product
    and computes the rest of the sum by a recursive relationship.

    Args:
        a_lo: linear operator.  implements __call__ for matvec
        vb: list of rhs vectors [v_0, ... v_p] where p is the highest order phi fn to compute.
        dt: step size.  TODO: support list of tau
        p: int. highest order phi fn to compute.
    """
    # TODO: this ONLY computes \tau=1.0 at this time, so does not support substepping
    tau = 1.0
    n = vb[0].shape[0]
    w = np.zeros((n,p+1))
    w[:,0] = vb[0]
    for j in range(1, p+1):
        for l in range(0, p-j+1):
            w[:, j] += (np.power(tau, l) / factorial(l)) * vb[j+l]
        w[:, j] += dt * a_lo(w[:, j-1])
    w_sum = np.zeros(n)
    for j in range(0, p-1+1):
        w_sum += (np.power(tau, j) / factorial(j)) * w[:, j]

    w_phi = phi_linop(a_lo, dt, w[:, p], p,
                      max_krylov_dim=max_krylov_dim, iom=iom)
    return w_phi + jnp.asarray(w_sum)
