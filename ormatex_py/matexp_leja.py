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

    coeffs = newton_poly_div_diff(leja_x,  jnp.exp(coeff*(shift + scale*leja_x)))
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
        _y_i = (dt*a_lo.matvec(y)/(scale)) + (y * (-shift/scale - leja_x[i-1]))
        y = y.at[:].set(_y_i)
        poly_expmv += coeffs[i] * y
        # estimate error
        poly_err = jnp.linalg.norm(y) * jnp.abs(coeffs[i])
        i += 1
        return i, y, poly_expmv, poly_err
    def cond_leja_poly(args):
        i, _, _, poly_err = args
        return (poly_err > tol) & (i < n_leja)
    i, y, poly_expmv, err = lax.while_loop(
            cond_leja_poly, body_leja_poly, (1, y, poly_expmv, 1.0e20))
    converged = i < n_leja
    # expmv, n_iters, converged
    return poly_expmv, i, converged


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
    eigs_a = np.linalg.eig(a)[0]
    max_abs_eig = np.max(np.abs(eigs_a))
    print("eigs:", eigs_a)
    # TODO: estimate largest eig by power iter
    alpha = -max_abs_eig
    shift = alpha / 2.
    scale = alpha / 4.
    a = jnp.asarray(a)
    a_lop = MatrixLinOp(a)
    u = jnp.ones(len(a))

    # calc exp(a_lop*dt)*u
    dt = 0.25
    lp = jnp.asarray(lp)
    leja_expmv, iter, converged = real_leja_expmv(a_lop, dt, u, 1.0, shift, scale, lp, 1e-6)
    expected_expmv = jax.scipy.linalg.expm(dt*a) @ u
    print(iter)
    print(converged)
    print(leja_expmv)
    print(expected_expmv)
