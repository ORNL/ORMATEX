"""
Matrix function evaluation using leja point
interpolation routines.

NOTE: When running on a CPU target, it is recommended to
use the following env variables for best performance:

    OMP_NUM_THREADS=4 XLA_FLAGS=--xla_cpu_use_thunk_runtime=false python rad_1d_3s.py <args>

Ref: https://github.com/jax-ml/jax/discussions/25711
"""
from functools import partial
import numpy as np
import jax
from jax import lax
from jax import numpy as jnp

# internal imports
from ormatex_py.ode_sys import LinOp, AugMatrixLinOp, MatrixLinOp, DiagLinOp


def gen_leja_conjugate(n: int=64, a: float=-1., b: float=1., c: float=1.):
    """
    Generate the conjugate leja points for an ellipse contained in the square
        [a,b] x [-c,c].

    This is the ellipse with center shift = ((a+b)/2, 0.) and half axes ((b-a)/2, c).

    Args:
        n: number of fast leja points to generate
        a: left boundary of box
        b: right boundary of box
        c: upper boundary of symmetric box
    """

    shift = (a+b)/2.
    hax1, hax2 = (b-a)/2., c
    scale = (hax1 + hax2) / 2

    # normalized half axes to capacity 1
    h1, h2 = hax1/scale, hax2/scale

    tol = 1e-4

    if h1 < 0 or h2 < 0:
        assert(False)
    elif h2 <= tol:
        # real points
        n_bigger = 2*(n - 1)
        ztc = gen_leja_circle(n_bigger, conjugate=True)
        zt = np.zeros(n)
        zt[:2] = np.real(ztc[:2])
        zt[2:n] = np.real(ztc[2:n_bigger:2])
        zt = h1 * zt
        n_real = n
    elif h1 <= tol:
        # imaginary points
        n_bigger = 2*(n - 1)
        ztc = gen_leja_circle(n_bigger, conjugate=True)
        zt = np.zeros(n, dtype=np.complex128)
        zt[:3] = np.imag(ztc[[0,2,3]])
        zt[3:n] = np.imag(ztc[4:n_bigger:2])
        zt = 1.j * h2 * zt
        n_real = 1
    else:
        zt = gen_leja_circle(n, conjugate=True)
        zt = h1 * np.real(zt) + 1j * h2 * np.imag(zt)
        n_real = 2

    return zt, n_real, scale, shift


def gen_leja_circle(n: int=64, conjugate=False):
    """
    Generate the fast leja points on a circle of radius 1 at zero.

    Ref:
        Baglama, J., D. Calvetti, and L. Reichel.
        "Fast leja points." Electron. Trans. Numer. Anal 7.124-140 (1998): 119-120.

    Args:
        n: number of fast leja points to generate
    """
    log2n = max(2, int(n-1).bit_length())
    nup = 2**log2n

    zt = np.zeros(nup, dtype=np.complex128)
    zt[0:2] = [+1, -1]

    for k in range(1, log2n):
        # multiply points by next root of unity to obtain the next 2^k points
        zt[2**k:2**(k+1)] = np.sqrt(zt[2**(k-1)]) * zt[:2**k]

    if conjugate:
        for k in range(1, log2n):
            # reorder points so they come in conjugate pairs
            mid_k = 2**k + 2**(k-1)
            zt[2**k:2**(k+1)] = np.vstack((zt[2**k:mid_k], zt[2**(k+1)-1:mid_k-1:-1])).flatten(order='F')

    return zt[:n]


def gen_leja_fast(a: float=-2., b: float=2., n: int=100):
    """
    Generate the fast leja points in [a, b].

    Ref:
        Baglama, J., D. Calvetti, and L. Reichel.
        "Fast leja points." Electron. Trans. Numer. Anal 7.124-140 (1998): 119-120.

    Args:
        a: left bounding point on the interval [a, b]
        b: right point
        n: number of fast leja points to generate
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


def gen_complex_conj_leja_fast(beta: float, n: int=100):
    r"""
    Generates the leja point sequence on the imaginary axis

    :math:`i[-\beta, \beta]`

    Args:
        beta: max imag eig magnitude
        n: number of fast leja points to generate
    """
    leja_x = gen_leja_fast(-beta, beta, n*2)
    # take only every other real leja point
    imag_even_leja_x = leja_x[0::2] * 1j
    # take complex conjugate of odd m points
    imag_odd_leja_x = leja_x[0::2] * -1j
    # rejoin full conj complex leja fast sequence
    leja_x = np.ones(len(leja_x), dtype=np.complex64) # can use np.complex128?
    leja_x[0::2] = imag_even_leja_x
    leja_x[1::2] = imag_odd_leja_x
    #leja_x[1::2] = imag_even_leja_x
    #leja_x[0::2] = imag_odd_leja_x
    leja_tmp = jnp.asarray(leja_x[0:n])
    # eliminate duplicate 0
    # TODO: Fix this. only works if interval is sym around 0
    # z0 should be 0
    return jnp.concat((leja_tmp[2:3], leja_tmp[0:2], leja_tmp[4:]))


@partial(jax.jit, static_argnums=(2,3))
def power_iter(a_lop: LinOp, b0: jax.Array, iter: int, tol: float=5.0e-3):
    """
    Performs power iteration to find dominant eigenvalue of system

    Args:
        a_lop:  LinOp linear operator which must implement matvec.
        b0: initial eigen vector
        iter: max number of power iterations to perform.
        tol: tolerance for dominant (real) eigenvalue
    """
    b_k = b0.at[:].get()
    eig_last = 1e20
    def body_power_iter(args):
        # unpack args: (i, bk, eig_a)
        i, b_k, eig_old, _ = args
        b_k1 = a_lop.matvec(b_k)
        eig_new = (b_k.transpose() @ b_k1) / \
                (b_k.transpose() @ b_k)
        b_k1_norm = jnp.linalg.norm(b_k1)
        # eig_new = b_k1_norm / jnp.linalg.norm(b_k)
        b_k = b_k1 / b_k1_norm
        i += 1
        return i, b_k, eig_new, eig_old
    def cond_power_iter(args):
        i, b_k, eig_new, eig_old = args
        eig_diff = jnp.abs(eig_new - eig_old)
        return (eig_diff > tol) & (i < iter)
    iters, b_k, eig_a, _ = lax.while_loop(
            cond_power_iter,
            body_power_iter,
            (0, b_k, eig_last, 1.0))
    return eig_a, b_k, iters


@jax.jit
def newton_poly_div_diff(x: jax.Array, y: jax.Array):
    """
    Divided difference formula to compute coeffs of the newton polynomial
    given pairs of (x_i, y_i)

    Args:
        x: data support points, where the data is sampled.
        y: value of the function at x points.
    """
    m = len(x)
    a = y.at[:].get() # TODO why not a = y?
    for k in range(1, m):
        a = a.at[k:m].set( (a[k:m] - a[k-1]) / (x[k:m] - x[k-1]) )
    return a


@jax.jit
def complex_conj_imag_leja_expmv(a_lo: LinOp, dt: float, u: jax.Array, shift: float, scale: float, imag_leja_x: jax.Array, coeffs: jax.Array, tol: float):
    r"""
    Interpolation approx :math:`\matrm{exp}(\delta t A)u` at the conjugate complex
    imaginary leja points.  The complex conj. leja points are symmetric about the real line,
    and admit a 2 term recurrenace relationship to compute the polynomial
    terms as described in

    Ref: Caliari, M., Kandolf, P., Ostermann, A. et al. Comparison of software for
        computing the action of the matrix exponential.
        Bit Numer Math 54, 113–128 (2014).
        https://doi.org/10.1007/s10543-013-0446-0

    Args:
        a_lo:  LinOp linear operator which must implement matvec.
        dt:  (Sub)step size.
        u: vector to which the matrix exponential is applied.
        shift:  shift applied to leja points.
        scale:  scale applied to leja points.
        imag_leja_x:  The unscaled complex leja sequence
    """
    print(f"jit-compiling complex_conj_imag_leja_expmv")
    n_leja = len(imag_leja_x)
    converged = False

    # norm of u for rel err est
    beta = jnp.linalg.norm(u)

    # leading const term in poly
    pm = jnp.real(coeffs[0]) * u

    # initial r
    rm = (dt*a_lo.matvec(u) - shift*u)/scale

    # storage vecs for recurrance
    qm = u.at[:].get() * 0.0

    # jax version of above
    def body_leja_poly(args):
        # unpack args
        i, rm, pm, qm, _ = args
        # compute error estimate
        err_r = jnp.real(coeffs[i-1])*jnp.linalg.norm(rm)
        # compute new term in the polynomial approx
        qm = (dt*a_lo.matvec(rm) - shift*rm)/scale
        # div diff coeffs[i] even is real
        # div diff coeffs[i-1] odd is complex
        pm = pm + jnp.real(coeffs[i-1])*rm + jnp.real(coeffs[i])*qm
        # update r
        rm = (dt*a_lo.matvec(qm) - shift*qm)/scale + jnp.pow(jnp.imag(imag_leja_x[i-1]), 2)*rm
        # estimate error correction
        poly_err = jnp.linalg.norm(qm) * jnp.abs(coeffs[i]) + err_r
        # jax.debug.print("i: {}, err: {}, coeffs[i]: {}, coeffs[i-1]: {}", i, poly_err, coeffs[i], coeffs[i-1])
        i += 2
        return i, rm, pm, qm, poly_err
    def cond_leja_poly(args):
        i, rm, pm, qm, poly_err = args
        # jax.debug.print("i: {}, h: {}, err: {}, scale: {}", i, dt, poly_err, scale)
        tol_check = (poly_err > tol*beta) & (poly_err < beta*1.0e3)
        iter_check = (i < n_leja)
        return (tol_check) & (iter_check) # | (i < 3)
    i, rm, pm, _qm, err = lax.while_loop(
            cond_leja_poly, body_leja_poly, (2, rm, pm, qm, 1.0e2))
    converged = (i < n_leja) & (err < beta*1.0e3)
    # expmv, n_iters, converged
    return pm, i, converged


@jax.jit
def decay_fun(e_new: float, e_old: float, gamma: float):
    return jnp.exp(gamma * jnp.log(e_new) + (1.-gamma) * jnp.log(e_old))


@partial(jax.jit, static_argnames='n_leja_real')
def complex_conj_leja_expmv(a_lo: LinOp, dt: float, u: jax.Array, shift: float, scale: float, leja_x: jax.Array, n_leja_real: int, coeffs: jax.Array, tol: float):
    r"""
    Interpolation approx :math:`\matrm{exp}(\delta t A)u` at real and conjugate complex
    leja points.

    leja_x[i] is real                for 0 <= i < n_leja_real
    leja_x[i+1] = conj(leja_x[i])    for i >= n_leja_real

    Note: this method does not check if leja_x has the assumed structure

    Args:
        a_lo:  LinOp linear operator which must implement matvec.
        dt:  (Sub)step size.
        u:  vector to which the matrix function is applied.
        shift:  shift applied to leja points.
        scale:  scale applied to leja points.
        leja_x:  unscaled complex leja sequence.
        n_leja_real: number of real leja points at the beginning of leja_x (one or two, usually).
        coeffs:  divied differences of the function applied to the matrix.
    """
    print(f"jit-compiling complex_conj_leja_expmv")
    n_leja = len(leja_x)
    if (n_leja_real == n_leja):
        return real_leja_expmv(a_lo, dt, u, shift, scale, leja_x, coeffs, tol)

    # apply scale and shift to leja points for readability
    leja_x_sc = shift + scale*leja_x

    # norm of u for error_estimate
    norm_u = jnp.linalg.norm(u)
    err_est = 2. * norm_u

    # decay coeff. for running average of err_est
    gamma = .5

    converged = (norm_u == 0.)

    # leading constant term in polynomial (real part for n_leja_real==0)
    vm = u
    pm = jnp.real(coeffs[0]) * vm

    # real leja points at the start
    # use a static loop and no tolerance check
    i = 1
    # while i <= n_leja_real:
    def body_real_leja_poly(args):
        i, vm, pm, err_est = args
        # compute new matvec (assume leja_x[i-1]) is real in exact arithmetic)
        vm = (dt*a_lo.matvec(vm) - jnp.real(leja_x_sc[i-1])*vm) / scale
        # apply update (for i==n_leja_real, coeff[i] is complex, we just compute the real part)
        pm += jnp.real(coeffs[i]) * vm
        err_est = decay_fun(jnp.abs(jnp.real(coeffs[i])) * jnp.linalg.norm(vm), err_est, gamma)
        # jax.debug.print("i: {}, err: {} coeffs[i]: {}", i, err_est, coeffs[i])
        i += 1
        return i, vm, pm, err_est
    def cond_real_leja_poly(args):
        i, _, _, err_est = args
        return (i <= n_leja_real)
    i, vm, pm, _ = lax.while_loop(cond_real_leja_poly, body_real_leja_poly, (i, vm, pm, 1.0e2))

    # jax version of the update for a conjugate part
    def body_leja_poly(args):
        # unpack args
        i, vm, pm, err_est, _ = args

        # compute new matvec (first one of conjugate pair, real part of complex matvec)
        qm = (dt*a_lo.matvec(vm) - jnp.real(leja_x_sc[i-1])*vm) / scale
        # real part of first update (coeff[i] is real in exact arithmetic)
        pm += jnp.real(coeffs[i]) * qm
        err_est = decay_fun(jnp.abs(jnp.real(coeffs[i])) * jnp.linalg.norm(qm), err_est, gamma)
        # jax.debug.print("i: {}, err: {} coeffs[i]: {}", i, err_est, coeffs[i])

        # compute new matvec (second one of conjugate pair, vm is real)
        vm = (dt*a_lo.matvec(qm) - jnp.real(leja_x_sc[i-1])*qm) / scale \
            + ((jnp.imag(leja_x_sc[i-1])/scale)**2)*vm
        # real part of second update (coeff[i+1] is complex, but vm real)
        pm += jnp.real(coeffs[i+1]) * vm
        norm_vm = jnp.linalg.norm(vm)
        err_est = decay_fun(jnp.abs(jnp.real(coeffs[i+1])) * norm_vm, err_est, gamma)
        # jax.debug.print("i: {}, err: {}, coeffs[i]: {}", i+1, err_est, coeffs[i+1])
        # jax.debug.print("i: {}, norm_vm: {}, err: {}, coeffs[i]: {}", i+1, norm_vm, err_est, coeffs[i+1])

        i += 2
        converged = err_est < tol*norm_u

        return i, vm, pm, err_est, converged
    def cond_leja_poly(args):
        i, _, _, err_est, converged = args
        cond = (i+1 < n_leja) & (err_est <= 1.e3*norm_u) & ~converged
        # jax.debug.print("i: {}, cond: {}, converged: {}", i, cond, converged)
        return cond
    i, _, pm, err_est, converged = lax.while_loop(
            cond_leja_poly, body_leja_poly, (i, vm, pm, err_est, converged))

    return pm, i, converged


@jax.jit
def real_leja_expmv(a_lo: LinOp, dt: float, u: jax.Array, shift: float, scale: float, leja_x: jax.Array, coeffs: jax.Array, tol: float):
    r"""
    Computes leja polynomial interpolation to approx :math:`\mathrm{exp}(\delta t A)u`.

    Ref:  L. Bergamaschi.  M. Caliari. A. Martinez and M. Vianello.
        Comparing Leja and Krylov Approximations of Large Scale
        Matrix Exponentials. Intl. Conf on Computational Science. 2006.

    Args:
        a_lo:  LinOp linear operator which must implement matvec.
        dt:  (Sub)step size.
        u: vector to which the matrix exponential is applied.
        shift:  shift applied to leja points.
        scale:  scale applied to leja points.
        leja_x:  The unscaled leja points
        tol:  Tolerance of the polynomial interpolation such that
            :math:`|| p - exp(\delta t A)u || < tol` where p is
            the leja polynomial approximation to :math:`\mathrm{exp}(\delta t A)u`.
    """
    print(f"jit-compiling leja_expmv")
    n_leja = len(leja_x)

    # apply scale and shift to leja points for readability
    leja_x_sc = shift + scale*leja_x

    # norm of u for error_estimate
    norm_u = jnp.linalg.norm(u)
    err_est = 2. * norm_u

    # decay coeff. for running average of err_est
    gamma = .5
    # decay_fun = lambda e_new, e_old: jnp.exp(gamma * jnp.log(e_new) + (1.-gamma) * jnp.log(e_old))

    converged = (norm_u == 0.)

    # leading constant term in polynomia
    pm = coeffs[0] * u
    vm = u

    def body_leja_poly(args):
        # unpack args
        i, vm, pm, err_est, _ = args

        # compute new matvec
        vm = (dt*a_lo.matvec(vm) - leja_x_sc[i-1]*vm) / scale
        # polynomial update
        pm += coeffs[i] * vm
        err_est = decay_fun(jnp.abs(coeffs[i]) * jnp.linalg.norm(vm), err_est, gamma)
        # jax.debug.print("i: {}, err: {} coeffs[i]: {}", i, err_est, coeffs[i])

        i += 1
        converged = err_est < tol*norm_u

        return i, vm, pm, err_est, converged
    def cond_leja_poly(args):
        i, _, _, err_est, converged = args
        cond = (i+1 < n_leja) & (err_est <= 1.e3*norm_u) & ~converged
        # jax.debug.print("i: {}, cond: {}, converged: {}", i, cond, converged)
        return cond
    i, _, pm, err_est, converged = lax.while_loop(
            cond_leja_poly, body_leja_poly, (1, vm, pm, err_est, converged))

    return pm, i, converged


def leja_shift_scale(a_tilde_lo: LinOp, dim: int, max_power_iter: int=20, b0=None, scale_factor: float=1.0):
    """
    Computes scaling and shifting parameters for leja interpolation
    of the matrix exponential using power iteration to estimate the
    eigenvalue with maximum real magnitude.
    TODO: use arnoldi or other method to estimate the eigenvalue with
    the maximum imag magnitude.
    WARNING: This routine does not work for systems with only
    imaginary eigenvalues.

    Args:
        max_power_iter: maximum number of power iterations.
        b0: vector. Initial guess for principle eigenvector of A.
        scale_factor: saftey factor for largest eig magnitude to
            ensure leja points encompass the spectrum.
    """
    if b0 is not None:
        assert len(b0) == dim
    else:
        b0 = jax.random.uniform(jax.random.key(42), (dim,))
    max_eig, b, iters = power_iter(a_tilde_lo, b0, max_power_iter)
    max_eig *= scale_factor
    alpha = max_eig
    beta = 0.  # assume min |eig(A)| is 0
    shift = (alpha - beta) / 2.
    scale = jnp.abs(beta - alpha) / 4.
    return shift, scale, max_eig, b, iters


@jax.jit
def leja_coeffs_exp_dd(leja_x: jax.Array, shift: float, scale: float, h: float=1.0):
    """
    Dividied difference computation of the leja polynomial coefficients.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
    """
    coeffs = newton_poly_div_diff(leja_x,  jnp.exp(h*(shift + scale*leja_x)))
    return coeffs


@jax.jit
def leja_coeffs_exp(leja_x: jax.Array, shift: float, scale: float, h: float=1.0):
    """
    Alternate method to compute the leja polynomial coefficients.
    Reduced roundoff error compared to divided difference formula.
    Ref:

        M. Calari.  Accurate evaluation of divided differences for polynomial
        interpolation of exponential propagators. Computing. 80. 2007.

    Args:
        leja_x:  The leja points
        shift:  The shift applied to the leja points
        scale:  The scale applied to the leja points
    """
    n_leja = len(leja_x)
    Xi = jnp.diag(leja_x, 0) + jnp.diag(jnp.ones(n_leja-1), -1)
    Xi_shift = jnp.identity(n_leja)*shift + scale*Xi
    # extract the first column
    coeffs = jax.scipy.linalg.expm(h*Xi_shift)[:, 0]
    return coeffs


def real_leja_expmv_substep(a_tilde_lo: LinOp, tau_dt: float, v: jax.Array, leja_x: jax.Array, n: int, shift: float, scale: float, tol: float=1.0e-10, substep: bool=True):
    r"""
    Computes :math:`exp(\Delta t A)v` using the real leja point
    interpolation method and substeps by:

    .. math::

        v_{i+1} = \mathrm{exp}(\tau_i \tilde A) v_{i}

    where :math:`\sum(\tau_i) = 1`

    Substepping is automatically applied by detecting divergence, or failure
    to converge the leja polynomial approximation.

    Args:
        a_tilde_lo:  LinOp linear operator.
        tau_dt: inital substep size in (0, 1].
        v: vector to which the matrix exponential is applied
        leja_x: leja points generated by the :func:`gen_leja_fast` function.
        substep: Flag to enable automatic substepping. True by default.
    """
    # assert tau_dt > 0
    # substep size
    dts = min(tau_dt, 1.0)
    # substep time
    tau = 0.0
    # current substep solution
    w_t = v
    tot_iter = 0
    i, max_substeps = 0, 1000
    max_tau_dt = 0.0
    last_converged = False

    # Compute leja polynomial coefficients
    coeffs = leja_coeffs_exp(leja_x, shift, scale)
    # coeffs = leja_coeffs_exp_dd(leja_x, shift, scale)

    # no substeps
    if not substep:
        w, iter, converged = real_leja_expmv(
                a_tilde_lo, 1.0, w_t, shift, scale, leja_x, coeffs, tol)
        return w[0:n], iter, converged, 1.0

    while True:
        w, iter, converged = real_leja_expmv(
                a_tilde_lo, dts, w_t, shift, scale, leja_x, coeffs, tol)
        tot_iter += iter
        # print(i, converged, tau, dts, iter, scale)
        print("sub_i: %d, converged: %d, sub_dt: %0.3f, iter: %d, shift: %0.3f, scale: %0.3f" % (i, converged, dts, iter, shift, scale))
        if not converged:
            # reduce the substep size
            dts /= 1.2
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
        if tau >= 1.0:
            break
        i += 1
        if i > max_substeps:
            raise RuntimeError("Max substeps reached")
    return w_t[0:n], tot_iter, converged, max_tau_dt


def complex_conj_leja_expmv_substep(a_tilde_lo: LinOp, tau_dt: float, v: jax.Array, leja_x: jax.Array, n_leja_real: int, n: int, shift: float, scale: float, tol: float=1.0e-10, substep: bool=True):
    r"""
    Computes :math:`exp(\Delta t A)v` using complex conjugate leja points and
    with substeps by:

    .. math::

        v_{i+1} = \mathrm{exp}(\tau_i \tilde A) v_{i}

    where :math:`\sum(\tau_i) = 1`

    Substepping is automatically applied by detecting divergence, or failure
    to converge the leja polynomial approximation.

    Args:
        a_tilde_lo:  LinOp linear operator.
        tau_dt: inital substep size in (0, 1].
        v: vector to which the matrix exponential is applied
        leja_x: leja points generated by the :func:`gen_leja_conjugate` function.
        n_leja_real: number of real points at the start of the leja sequence.
          This value is returned by :func:`gen_leja_conjugate`.
          `n_leja_real==2` for a circle or ellipse, but can be equal to len(leja_x)
          in the case of purely real leja points.  If n_leja_real is equal to
          len(leja_x) then this method is equal to the :func:`real_leja_expmv_substep` method.
        substep: Flag to enable automatic substepping. True by default.
    """
    # assert tau_dt > 0
    # substep size
    dts = min(tau_dt, 1.0)
    # substep time
    tau = 0.0
    # current substep solution
    w_t = v
    tot_iter = 0
    i, max_substeps = 0, 1000
    max_tau_dt = 0.0
    last_converged = False

    # Compute leja polynomial coefficients
    coeffs = leja_coeffs_exp(leja_x, shift, scale)
    # coeffs = leja_coeffs_exp_dd(leja_x, shift, scale)

    # no substeps
    if not substep:
        w, iter, converged = complex_conj_leja_expmv(
                a_tilde_lo, 1.0, w_t, shift, scale, leja_x, n_leja_real, coeffs, tol)
        return w[0:n], iter, converged, 1.0

    while True:
        w, iter, converged = complex_conj_leja_expmv(
                a_tilde_lo, dts, w_t, shift, scale, leja_x, n_leja_real, coeffs, tol)
        tot_iter += iter
        print("sub_i: %d, converged: %d, sub_dt: %0.3f, iter: %d, shift: %0.3f, scale: %0.3f" % (i, converged, dts, iter, shift, scale))
        if not converged:
            # reduce the substep size
            dts /= 1.2
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
        if tau >= 1.0:
            break
        i += 1
        if i > max_substeps:
            raise RuntimeError("Max substeps reached")
    return w_t[0:n], tot_iter, converged, max_tau_dt


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


def example_fast_leja_points():
    import matplotlib.pyplot as plt
    # generate leja points
    lp = gen_leja_fast(a=-2, b=2, n=50)
    # first 10 leja points
    for v in lp[0:50]:
        print(v)

    # plot leja points on the complex plane
    plt.figure()
    cmap = plt.get_cmap('rainbow')
    plt.scatter(np.real(lp), np.imag(lp), c=list(range(0, len(lp))), cmap=cmap)
    plt.title("Fast real Leja sequence")
    plt.grid(ls='--')
    plt.xlabel("Re")
    plt.ylabel("Im")
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

    scale_factor = 1.01
    alpha = max_eig * scale_factor
    print("alpha a:", alpha)
    shift = alpha / 2.
    scale = alpha / 4.

    # calc exp(a_lop*dt)*u
    dt = 0.25
    lp = jnp.asarray(lp)
    coeffs = newton_poly_div_diff(lp,  jnp.exp(shift + scale*lp))
    leja_expmv, iter, converged = real_leja_expmv(a_lop, dt, u, shift, scale, lp, coeffs, 1e-6)
    expected_expmv = jax.scipy.linalg.expm(dt*a) @ u
    print(iter)
    print(converged)
    print("real leja expmv: ", leja_expmv)
    print("true expmv:" , expected_expmv)

    # imag eigs only
    a = np.array([[0.0, 2.5],
                  [-2.5, 0.0]])
    a_lop = MatrixLinOp(a)
    dim = len(a)
    eigs_a = np.linalg.eig(a)[0]
    a = jnp.asarray(a)
    print("eigs a:", eigs_a)

    u = jnp.ones(len(a))
    # Estimate largest eig by power iter
    b0 = jax.random.uniform(jax.random.key(42), (dim,))
    max_eig, _, _ = power_iter(a_lop, b0, 10)

    scale_factor = 1.01
    alpha = max_eig * scale_factor
    print("alpha a:", alpha)

    # generate imag leja sequence on interval i[-2, 2]
    ip = gen_complex_conj_leja_fast(2, n=40)
    plt.figure()
    cmap = plt.get_cmap('rainbow')
    plt.scatter(np.real(ip), np.imag(ip), c=list(range(0, len(ip))), cmap=cmap)
    plt.title("Conjugate complex fast Leja sequence")
    plt.grid(ls='--')
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.savefig('fast_imag_leja_points.png')
    plt.grid()
    plt.close()

    # compute expmv of matrix a with imaginary spectrum:  exp(a*dt)*u
    dt = 1.0
    expected_expmv = jax.scipy.linalg.expm(dt*a) @ u
    # max extent of the matrix spectrum on imag axis
    beta = np.abs(np.max(np.imag(eigs_a)))

    # compute polynomial coeffs by div diff for complex conj leja sequence
    shift = 0.
    scale = beta / 4.
    coeffs_dd = leja_coeffs_exp_dd(ip, shift, scale)
    coeffs_exp = leja_coeffs_exp(ip, shift, scale)
    print("===complex conj leja===")
    print(ip)
    print("===div diffs===")
    # print(coeffs_dd)
    print(coeffs_exp)
    leja_expmv, iter, converged = complex_conj_imag_leja_expmv(
            a_lop, dt, u, shift, scale, ip, coeffs_exp, 1e-6)
    print("conj complex leja expmv:", leja_expmv)
    print("true expmv:", expected_expmv)


def example_leja_conjugate_points():
    import matplotlib.pyplot as plt
    a = -2. # use 0 for imaginary interval
    c = 10.
    lpc, _, scalec, shiftc = gen_leja_conjugate(n=26, a=a, b=0., c=c)
    print([scalec, shiftc])
    print(lpc[:,None])

    # plot leja points on the complex plane
    plt.figure()
    cmap = plt.get_cmap('rainbow')
    plt.scatter(np.real(lpc), np.imag(lpc), c=list(range(0, len(lpc))), cmap=cmap)
    plt.grid(ls='--')
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Conjugate complex Leja points on ellipse \n a: %0.2f, c: %0.2f, shift: %0.2f, scale: %0.2f" % (a, c, shiftc, scalec))
    plt.savefig('leja_points_ellipse.png')
    plt.grid()
    plt.close()

    lpc = gen_leja_circle(n=26, conjugate=True)
    # plot leja points on the complex plane
    plt.figure()
    cmap = plt.get_cmap('rainbow')
    plt.scatter(np.real(lpc), np.imag(lpc), c=list(range(0, len(lpc))), cmap=cmap)
    plt.grid(ls='--')
    plt.title("Conjugate complex Leja points on circle")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.savefig('leja_points_circle.png')
    plt.grid()
    plt.close()


def example_leja_conjugate_ellipse_error(a=0., b=0., c=4.):
    # a=0 for imaginary interval
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=3)

    #lpc = gen_leja_circle(n=20, conjugate=True)
    n_max = 21 # 32
    lp, n_leja_real, scale, shift = gen_leja_conjugate(n=n_max, a=a, b=b, c=c)

    # plot leja points on the complex plane
    plt.figure()
    cmap = plt.get_cmap('rainbow')
    plt.scatter(np.real(lp), np.imag(lp), c=list(range(0, len(lp))), cmap=cmap)
    plt.grid(ls='--')
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Leja conjugate ellipse \n a: %0.2f, c: %0.2f, shift: %0.2f, scale: %0.2f" % (a, c, shift, scale))
    plt.savefig('leja_points_conjugate_ellipse_a%0.2f.png' % a)
    plt.grid()
    plt.close()

    # generate a diagonal matrix and wrap as a linear operator
    xr_grid = jnp.linspace(-3,1,100)
    xi_grid = jnp.linspace(-6,6,100)
    zr_grid, zi_grid = jnp.meshgrid(xr_grid, xi_grid)

    zs = zr_grid.flatten() + 1.j * zi_grid.flatten()

    a_lop = DiagLinOp(zs)
    u = jnp.ones(zs.shape, dtype=jnp.complex128)

    # calc exp(a_lop) * u
    expected_expmv = jnp.exp(zs)
    lp = jnp.asarray(lp)

    if scale < 1e-1:
        new_scale = np.sqrt(3.**2 + 6.**2)/2
        lp = lp * scale / new_scale
        scale = new_scale

    # compute polynomial coeffs by div diff for complex conj leja sequence
    print([scale, shift])
    coeffs_dd = leja_coeffs_exp_dd(lp, shift, scale)
    coeffs_exp = leja_coeffs_exp(lp, shift, scale)
    print("===div diffs===")
    print(np.hstack((lp[:,None], coeffs_exp[:,None], coeffs_dd[:,None])))
    # shift, scale = 0., 1.0
    leja_expmv, i, converged = complex_conj_leja_expmv(
            a_lop, 1., u, shift, scale, lp, n_leja_real, coeffs_exp, 1e-2)
    print([int(i), bool(converged)])
    print("conj complex leja expmv:", leja_expmv)
    print("true expmv:", expected_expmv)

    err = jnp.abs(expected_expmv-leja_expmv)
    print("error:", err)

    plt.figure()
    plt.contourf(jnp.real(zs.reshape(zr_grid.shape)),
                 jnp.imag(zs.reshape(zr_grid.shape)),
                 jnp.log(err.reshape(zr_grid.shape))/jnp.log(10.), vmin=-7, extend='both')
    plt.title("Leja interpolation error $\\mathrm{log}|e^{Z}u - p(Z)_{leja}|$ error \n a: %0.2f, c: %0.2f, shift: %0.2f, scale: %0.2f" % (a, c, shift, scale))
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.colorbar()
    plt.scatter(np.real(shift + scale*lp[:i]), np.imag(shift+ scale*lp[:i]), c='r', s=8)
    plt.grid(ls='--')
    plt.savefig('leja_points_conjugate_ellipse_error_a%0.2f.png' % a, dpi=120)
    plt.close()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    example_fast_leja_points()
    example_leja_conjugate_points()
    example_leja_conjugate_ellipse_error(a=-2.0, c=3.5)
    example_leja_conjugate_ellipse_error(a=-2.5, b=.5, c=5.5)
    example_leja_conjugate_ellipse_error(a=0, b=0, c=6)
    example_leja_conjugate_ellipse_error(a=-3, b=1, c=0)
    example_leja_conjugate_ellipse_error(a=-1e-5, b=0, c=1e-5)
    example_leja_conjugate_ellipse_error(a=-2.99, b=.99, c=.01)
