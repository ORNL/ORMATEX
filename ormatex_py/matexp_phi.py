"""
Implements the phi-functions
"""
from functools import partial
import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx

import warnings

def f_phi_k(z: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z) for dense Z
    """
    mach_eps = jnp.finfo(z.dtype).eps
    phi_k, err_est = f_phi_k_inv(z, k=k, eps=mach_eps)

    if err_est > np.sqrt(mach_eps):
        warnings.warn("the argument of phi_k is close to singular, and f_phi_k is inaccurate.\n" \
                      f"estimated error={err_est}. use f_phi_k_ext instead.")

    return phi_k

def _validate_args(z: jax.Array, k: int):
    assert k >= 0
    N, M = z.shape
    assert N == M
    return N

@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_inv(z: jax.Array, k: int, eps: float) -> (jax.Array, float):
    """
    Computes phi_k(Z) for dense Z, using a formula involving the inverse of Z.
    Returns an error estimate to warn about nearly singular Z.
    """
    N = _validate_args(z, k)

    # phi_0 = exp
    phi_k = jax.scipy.linalg.expm(z)
    err_est = eps
    if k == 0:
        return phi_k, err_est
    else:
        # compute a qr decomposition instead of inverse
        Qz, Rz = jax.scipy.linalg.qr(z)
        r_min = jnp.min(jnp.abs(jnp.diag(Rz)))
        I = jnp.eye(N)
        #DEBUG
        #jax.debug.print("Qz: {Q}, Rz: {R}, r_min: {r_min}", Q=Qz, R=Rz, r_min=r_min)
        #jax.debug.print("phi_0: {M}", M=phi_k)
        for ki in range(1, k+1):
            kfact = 1./jax.scipy.special.factorial(ki-1)
            phi_k = jax.scipy.linalg.solve_triangular(Rz, Qz.T @ (phi_k - kfact*I))
            err_est = err_est / r_min

            #jax.debug.print("phi_k: {M}, err_est: {err_est}", M=phi_k, err_est=err_est)
    return phi_k, err_est

@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_ext(z: jax.Array, k: int, return_all: bool=False) -> jax.Array:
    """
    Computes phi_k(Z) for dense Z, using the stable but more expensive extension formula
    """
    N = _validate_args(z, k)

    if k > 0:
        z_ext_k = jnp.block([[z], [jnp.zeros(((k-1)*N, N))]])
        z_ext = jnp.block([[z_ext_k, jnp.eye(k*N)], [jnp.zeros((N, (k+1)*N))]])
    else:
        z_ext = z

    phi_ks = jax.scipy.linalg.expm(z_ext)

    if return_all:
        phi_k = phi_ks[:N,:].reshape((N,k+1,N))
        phi_k = phi_k.swapaxes(0, 1) # swap the k index to first axes
        return phi_k
    else:
        phi_k = phi_ks[:N,-N:]
        return phi_k

@partial(jax.jit, static_argnums=(1,2))
def f_phi_k_poly_all(z: jax.Array, k: int, poly_deg: int=4) -> list[jax.Array]:
    """
    Computes phi_k(Z) for dense Z, using a Taylor polynomial
    """
    N = _validate_args(z, k)
    assert(poly_deg >= k+1)

    fact_k = jax.scipy.special.factorial(k)
    zpow_kfac = z / fact_k / (k+1.)
    poly_approx = jnp.eye(N)/fact_k + zpow_kfac

    for j in range(k+2, poly_deg+1):
        zpow_kfac = z @ zpow_kfac / j
        poly_approx = poly_approx + zpow_kfac

    phi_ks = [None] * (k+1)
    phi_ks[k] = poly_approx
    fact_j = fact_k / k
    for j in range(k-1, -1, -1):
        # recursion formula phi_j(z) = 1/j! + z phi_{j+1}(z)
        phi_ks[j] = jnp.eye(N)/fact_j + z @ phi_ks[j+1]
        fact_j = fact_j / j

    return phi_ks

@partial(jax.jit, static_argnums=(1,))
def f_phi_k_sq_all(z: jax.Array, k: int) -> list[jax.Array]:
    """
    Computes phi_k(Z) for dense Z, using the scaling and squaring relations
    """
    N = _validate_args(z, k)

    # use infty matrix norm instead of spectral radius to determine scaling
    theta = jnp.linalg.norm(z, ord=np.inf)
    # TODO: determine the optimal initial polynomial degree and the number of squarings
    scale_fact = 16
    init_poly_deg = 4
    Nscale = jnp.floor(jnp.maximum(0, jnp.log2(theta * scale_fact))).astype(int)
    tt_N = 2 ** Nscale

    # compute the initial approximation of the phi functions for scaled z
    phi_ks = f_phi_k_poly_all(z / tt_N, k, poly_deg=init_poly_deg)

    # determine scaling constants
    zero_to_k = jax.lax.iota(z.dtype, k+1)
    scalings = 1. / (jax.scipy.special.factorial(zero_to_k[:,None] - zero_to_k[None,:]) * 2. ** zero_to_k[:,None])

    #jax.debug.print("scalings={a}, Nscale={b}", a=scalings, b=Nscale)

    def sq_step(counter, phi_ks):
        for j in range(k, 0, -1):
            # first term in the sum and last correction
            phi_ks[j] = (phi_ks[0] @ phi_ks[j] + phi_ks[j]) * scalings[j,j]
            for jj in range(j-1, 0, -1):
                # remaining corrections
                phi_ks[j] = phi_ks[j] + phi_ks[jj] * scalings[j,jj]
        # traditional squaring of exp
        phi_ks[0] = phi_ks[0] @ phi_ks[0]
        return phi_ks

    phi_ks = jax.lax.fori_loop(0, Nscale, sq_step, phi_ks)

    return phi_ks

def f_phi_k_sq(z: jax.Array, k: int, return_all: bool=False) -> jax.Array:

    phi_ks = f_phi_k_sq_all(z, k)
    if return_all:
        return jnp.array(phi_ks)
    else:
        return phi_ks[k]

## methods for phi_k(A)B

def _validate_args_appl(z: jax.Array, b: jax.Array, k: int):
    assert k >= 0
    assert len(z.shape) == 2
    N, N1 = z.shape
    assert N == N1
    if len(b.shape) == 1:
        N2 = b.shape[0]
        M = 1
        B = b[:,None]
        assert N2 == N
    else:
        assert len(b.shape) == 2
        N2, M = b.shape
        B = b
        assert N2 == N

    return N, M, B

@partial(jax.jit, static_argnums=(2,))
def f_phi_k_appl(z: jax.Array, b: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z)B for dense Z and dense B, using an extension formula
    """
    N, M, B = _validate_args_appl(z, b, k)

    if k > 0:
        z_ext = jnp.block([ [z, B, jnp.zeros((N, (k-1)*M))],
                            [jnp.zeros(((k-1)*M, N+M)), jnp.eye((k-1)*M)],
                            [jnp.zeros((M, N+k*M))] ])
        # phi_k_ext = jax.scipy.linalg.expm(z_ext, max_squarings=20)
        phi_k_ext = f_phi_k_sq(z_ext, k=0)
        phi_kb = phi_k_ext[:N,-M:].reshape(b.shape)
    else:
        phi_kb = jax.scipy.linalg.expm(z) @ b

    return phi_kb

@partial(jax.jit, static_argnums=(2,))
def f_phi_k_pole(z: jax.Array, b: jax.Array, k: int) -> jax.Array:
    """
    Computes phi_k(Z)B for dense Z and dense B, using a rational approximation and partial fraction expansion
    """
    N, M, B = _validate_args_appl(z, b, k)

    # poles
    method = 'CN' #'cram_16'
    ps, cs, c0 = _pole_dict[method]

    phi_kb = jnp.zeros(B.shape)
    Id = jnp.eye(z.shape[0])

    if k == 0:
        phi_kb += c0 * B

    for p, c in zip(ps, cs):
        phi_kb += jnp.real((2. * c / p**k) * jnp.linalg.solve((z - p*Id), B))

    return phi_kb.reshape(b.shape)

_pole_dict = {
    'IE': ([1.], [-1/2.], 0.),
    'CN': ([2.], [-4./2.], -1.),
    '2': ([0.18676273583571287,
           1.3048745815331186],
          [ 0.020796803046292*0.1867627358357128/2,
           -1.443886338474624*1.3048745815331186/2],
          0.),
    '3': ([ 0.09001193971998,
            0.37446800561543,
            1.02640247374119 ],
          [ -0.0167957123140054*0.09001193971998/2,
             0.2817994107299859*0.37446800561543/2,
            -1.6499133268977124*1.02640247374119/2 ],
          0.),
    '6': ([ 0.042426068840,
            0.101993391742,
            0.281871588952,
            0.700834344988,
            1.495681999245,
            3.820326308790 ],
          [ -0.00143004020*0.042426068840/2,
             0.01492008680*0.101993391742/2,
            -0.14717772455*0.281871588952/2,
             1.12526293779*0.700834344988/2,
            -4.26534836471*1.495681999245/2,
             2.50083847742*3.820326308790/2 ],
          0.),
    '7': ([ 0.04443232806036,
            0.10011808797191,
            0.25566954775280,
            0.59034307631118,
            1.17013664097710,
            2.48317240906630,
            8.09785048326380 ],
          [-0.00408266990779/2,
            0.04316003633844/2,
           -0.41908515130620/2,
            3.09546382847179/2,
          -10.86914678824066/2,
           12.21215072372589/2,
           -4.65569796438472/2 ],
          0.),
    'cram_16': (
         [ -10.843917078696988026 + 19.277446167181652284j,
           -5.2649713434426468895 + 16.220221473167927305j,
            5.9481522689511774808 + 3.5874573620183222829j,
            3.5091036084149180974 + 8.4361989858843750826j,
            6.4161776990994341923 + 1.1941223933701386874j,
            1.4193758971856659786 + 10.925363484496722585j,
            4.9931747377179963991 + 5.9968817136039422260j,
           -1.4139284624888862114 + 13.497725698892745389j ],
         [ -0.0000005090152186522491565 + -0.00002422001765285228797j,
              0.00021151742182466030907 +   0.0043892969647380673918j,
                  113.39775178483930527 +       101.9472170421585645j,
                  15.059585270023467528 +     -5.7514052776421819979j,
                 -64.500878025539646595 +     -224.59440762652096056j,
                 -1.4793007113557999718 +      1.7686588323782937906j,
                 -62.518392463207918892 +      -11.19039109428322848j,
                0.041023136835410021273 +    -0.15743466173455468191j ],
        2.1248537104952237488e-16 )
    }
