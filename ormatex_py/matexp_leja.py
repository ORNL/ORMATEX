"""
Matrix function evaluation using leja point
interpolation routines.
"""
from functools import partial
import numpy as np
import jax
from jax import lax
from jax import numpy as jnp

# internal imports
from ormatex_py.ode_sys import LinOp, AugMatrixLinOp, MatrixLinOp


def gen_leja_fast(a=-2, b=2., n=100):
    """
    Generate the fast leja points.  Ref:
        Baglama, J., D. Calvetti, and L. Reichel.
        "Fast leja points." Electron. Trans. Numer. Anal 7.124-140 (1998): 119-120.
    """
    # the first 3 fast leja points
    zt = np.zeros(n)
    zt[0:3] = [a, b, (a+b)/2.] if abs(a) > abs(b) else [b, a, (a+b)/2.]
    # canidate points
    zs = np.zeros(n)
    zs[0] = (zt[1]+zt[2])/2.
    zs[1] = (zt[2]+zt[0])/2.
    zprod = np.zeros(n)
    zprod[0] = np.prod(zs[0]- np.asarray(zt))
    zprod[1] = np.prod(zs[1]-np.asarray(zt))
    index = np.zeros((n,2), dtype=int)
    index[0,0] = 1
    index[0,1] = 2
    index[1,0] = 2
    index[1,1] = 0
    for i in range(3, n):
        maxi = np.argmax(np.abs(zprod))
        zt[i] = zs[maxi]
        # zt.append( zs[maxi] )
        index[i-1, 0] = i
        index[i-1, 1] = index[maxi, 1]
        index[maxi,1] = i

        zs[maxi] = (zt[index[maxi,0]]+zt[index[maxi,1]])/2.
        zs[i-1] = (zt[index[i-1,0]]+zt[index[i-1,1]])/2.

        zprod[maxi] = np.prod(zs[maxi]-zt[0:i])
        zprod[i-1] = np.prod(zs[i-1]-zt[0:i])
        zprod = np.asarray(zprod)*(zs-zt[i])
    return zt


def gen_leja(n=100):
    """
    Generate leja points

    Args:
        n: maximum number of leja points to generate
    """
    # TODO
    pass


@partial(jax.jit, static_argnums=(2))
def power_iter(a_lop: LinOp, b0: jax.Array, iter: int):
    """
    Performs power iteration to find dominate eigenvalue of system
    """
    # b_k = jax.random.uniform(jax.random.key(42), (dim,))
    # inital guess for eigenvector
    tol = 1.0e-3
    b_k = b0.at[:].get()
    eig_last = 1e20
    # for _ in range(iter):
    #     b_k1 = a_lop.matvec(b_k)
    #     eig_a = (b_k.transpose() @ b_k1) / \
    #             (b_k.transpose() @ b_k)
    #     b_k1_norm = jnp.linalg.norm(b_k1)
    #     b_k = b_k1 / b_k1_norm
    #     if eig_diff < tol:
    #         break
    #     eig_last = eig_a
    # jax version of above
    def body_power_iter(args):
        # unpack args: (i, bk, eig_a)
        i, b_k, eig_old, _ = args
        b_k1 = a_lop.matvec(b_k)
        eig_new = (b_k.transpose() @ b_k1) / \
                (b_k.transpose() @ b_k)
        b_k1_norm = jnp.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        i += 1
        return i, b_k, eig_new, eig_old
    def cond_power_iter(args):
        i, b_k, eig_new, eig_old = args
        eig_diff = jnp.abs(eig_new - eig_old)
        return (eig_diff > tol) & (i < iter)
    iters, b_k, eig_a, _ = lax.while_loop(cond_power_iter, body_power_iter, (0, b_k, eig_last, 1.0))
    return eig_a, b_k, iters


@jax.jit
def newton_poly_div_diff(x: jax.Array, y: jax.Array):
    """
    Divided difference formula to compute coeffs of the newton polynomial
    given pairs of (x_i, y_i)
    """
    m = len(x)
    a = y.at[:].get()
    for k in range(1, m):
        a = a.at[k:m].set( (a.at[k:m].get() - a.at[k - 1].get())/(x.at[k:m].get() - x.at[k - 1].get()) )
    return a


@jax.jit
def real_leja_expmv(a_lo: LinOp, dt: float, u: jax.Array, coeff: float, shift: float, scale: float, leja_x: jax.Array, tol: float):
    """
    Computes leja polynomial interpolation to approx exp(dt*A)*v

    Args:
        shift:  shift applied to leja points.  q=(1/2)*alpha.  Where alpha=0.5*max(abs(eigs(A))
        scale:  scale applied to leja points.
    """
    print(f"jit-compiling leja_expmv")
    n_leja = len(leja_x)
    converged = False

    coeffs = newton_poly_div_diff(leja_x,  jnp.exp(coeff*dt*(shift + scale*leja_x)))
    # coeffs = newton_poly_div_diff(leja_x,  jnp.exp(coeff*(leja_x)))
    # polynomial approximation, 1st term
    poly_expmv = coeffs[0] * u
    y = u.at[:].get()

    # for i in range(1, n_leja):
    #     y_i = (dt*a_lo.matvec(y)/(scale)) + (y * (-shift/scale - leja_x[i-1]))
    #     y = y.at[:].set(y_i)
    #     poly_expmv += coeffs[i] * y
    #     # estimate error
    #     poly_err = jnp.linalg.norm(y) * jnp.abs(coeffs[i])
    #     if poly_err < tol:
    #         converged = True
    #         break
    # jax version of above
    def body_leja_poly(args):
        # unpack args: (i, y, poly, err)
        i, y, poly_expmv, _ = args
        # compute new term in the polynomial approx
        _y_i = (a_lo.matvec(y)/(scale)) + (y * (-shift/scale - leja_x[i-1]))
        y = y.at[:].set(_y_i)
        poly_expmv += coeffs[i] * y
        # estimate error
        poly_err = jnp.linalg.norm(y) * jnp.abs(coeffs[i])
        i += 1
        return i, y, poly_expmv, poly_err
    def cond_leja_poly(args):
        i, y, poly_expmv, poly_err = args
        tol_check = (poly_err > tol*jnp.linalg.norm(poly_expmv)+tol) & (i < n_leja) & (poly_err < 2.0e4)
        return tol_check
    i, y, poly_expmv, err = lax.while_loop(
            cond_leja_poly, body_leja_poly, (1, y, poly_expmv, 1.0e4))
    converged = (i < n_leja) & (err < 2.0e4)
    # expmv, n_iters, converged
    return poly_expmv, i, converged

def leja_phikv_extended(
        a_tilde_lo: LinOp, dt: float, v: jax.Array, leja_x: jax.Array,
        n: int, shift: float, scale: float,
        n_substeps: int=1, tol: float=1.0e-6) -> jax.Array:
    w, iter, converged = real_leja_expmv(a_tilde_lo, dt, v, 1.0,
              shift, scale, leja_x, tol)
    return w[0:n], iter, converged

def leja_shift_scale(a_tilde_lo: LinOp, dim: int, max_power_iter: int=20, b0=None, scale_factor: float=1.1):
    """
    Args:
        max_power_iter: maximum number of power iterations
        b0: vector. Initial guess for principle eigenvector of A
        scale_factor: saftey factor for largest eig magnitude to ensure leja points
            encompass the system jacobian spectrum
    """
    if b0 is not None:
        assert len(b0) == dim
    else:
        b0 = jax.random.uniform(jax.random.key(42), (dim,))
    max_eig, b, iters = power_iter(a_tilde_lo, b0, max_power_iter)
    alpha = max_eig * scale_factor
    shift = alpha / 2.
    scale = alpha / 4.
    return shift, scale, max_eig, b, iters

def real_leja_expmv_substep(a_tilde_lo, tau_dt, v, leja_x, n, shift, scale, tol=1.0e-6):
    """
    Computs exp(dt*A)*v using substeps by:
        v_{i+1} = exp(dt*A*tau_i)*v_{i}
        where sum(tau_i) = 1
    Args:
        tau_dt: inital substep size in (0, 1].
    """
    assert tau_dt > 0
    # substep size
    dts = min(tau_dt, 1.0)
    # substep time
    tau = 0.0
    # current substep solution
    w_t = v
    tot_iter = 0
    i, max_substeps = 0, 200
    max_tau_dt = 0.0
    last_converged = False
    while True:
        w, iter, converged = real_leja_expmv(
                a_tilde_lo, dts, w_t, 1.0, shift, scale, leja_x, tol)
        tot_iter += iter
        if not converged:
            # reduce the substep size
            dts /= 2.
            last_converged = False
        else:
            # the maximum accepted step size
            max_tau_dt = max(max_tau_dt, dts)
            # accept the substep
            tau = tau + dts
            w_t = w
            # clip substep size
            dts = min(dts, 1.0 - tau)
            last_converged = True
        # print(i, converged, tau, dts, iter)
        if tau >= 1.0:
            break
        i += 1
        if i > max_substeps:
            raise RuntimeError("Max substeps reached")
    return w_t[0:n], tot_iter, converged, max_tau_dt


def leja_phikv_extended_substep(a_tilde_lo, tau_dt, v, leja_x, n, shift, scale, tol):
    w, tot_iters, converged, dts = real_leja_expmv_substep(
            a_tilde_lo, tau_dt, v, leja_x, n, shift, scale, tol)
    return w, tot_iters, converged, dts


def build_a_tilde(a_lo: LinOp, dt: float, vb: list[jax.Array]):
    """
    Builds extended linear system.

    Args:
        a_lo:  System Linear operator
        dt: time step size
        vb: list of rhs vectors to which :math:`A` will be applied to

    Builds the matrix:

    .. math::

        \tilde A = [[A, B],[0, K]]

    where A = a_lo is NxN,
    B = vb[:0:-1] is Nxp,
    K = [[0, I_{p-1}],[0, 0]] is pxp,
    :math:` \tilde A ` is N+p x N+p
    """
    p = len(vb) - 1
    n = vb[0].shape[0]

    # build B
    b = jnp.vstack(vb[:0:-1]).T

    # build \tilde A
    k = np.zeros((p,p))
    k[0:p-1, 1:] = np.eye(p-1)
    k = jnp.asarray(k)
    a_tilde_lo = AugMatrixLinOp(a_lo, dt, b, k)

    unit_vec = np.zeros(p)
    unit_vec[-1] = 1.0
    v = jnp.concat((vb[0], jnp.asarray(unit_vec)))
    return a_tilde_lo, v, n


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # generate leja points
    lp = gen_leja_fast(a=-2, b=2, n=20)
    # first 10 leja points
    for v in lp[0:10]:
        print(v)

    # plot leja points on the complex plane
    plt.figure()
    cmap = plt.get_cmap('rainbow')
    plt.scatter(np.real(lp), np.imag(lp), c=list(range(0, len(lp))), cmap=cmap)
    plt.savefig('fast_leja_points.png')
    plt.grid()
    plt.close()

    # generate a matrix and wrap as a linear operator
    a = np.array([[-1.25, 0.0],
                  [1.25, -0.5]])
    dim = len(a)
    eigs_a = np.linalg.eig(a)[0]
    a = jnp.asarray(a)
    print("eigs a:", eigs_a)

    a_lop = MatrixLinOp(a)
    u = jnp.ones(len(a))
    # Estimate largest eig by power iter
    b0 = jax.random.uniform(jax.random.key(42), (dim,))
    max_eig, _, _ = power_iter(a_lop, b0, 10)

    scale_factor = 1.1
    alpha = max_eig * scale_factor
    print("alpha a:", alpha)
    shift = alpha / 2.
    scale = alpha / 4.

    # calc exp(a_lop*dt)*u
    dt = 0.25
    lp = jnp.asarray(lp)
    leja_expmv, iter, converged = real_leja_expmv(a_lop, dt, u, 1.0, shift, scale, lp, 1e-6)
    expected_expmv = jax.scipy.linalg.expm(dt*a) @ u
    print(iter)
    print(converged)
    print(leja_expmv)
    print(expected_expmv)

    # imag eigs only
    a = np.array([[0.0, 1.0],
                  [-1.0, 0.0]])
    dim = len(a)
    eigs_a = np.linalg.eig(a)[0]
    a = jnp.asarray(a)
    print("eigs a:", eigs_a)

    a_lop = MatrixLinOp(a)
    u = jnp.ones(len(a))
    # Estimate largest eig by power iter
    b0 = jax.random.uniform(jax.random.key(42), (dim,))
    max_eig, _, _ = power_iter(a_lop, b0, 10)

    scale_factor = 1.1
    alpha = max_eig * scale_factor
    print("alpha a:", alpha)
