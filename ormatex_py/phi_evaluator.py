"""
Exponential time integration methods
"""
import jax
import jax.numpy as jnp
import numpy as np

from abc import abstractmethod
import equinox as eqx

from functools import partial

from ormatex_py.ode_sys import LinOp

from ormatex_py.matexp_krylov import phi_linop, kiops_fixedsteps
from ormatex_py.matexp_phi import (
    f_phi_k_sq_all,
    f_phi_ks_pfd,
    get_pfd_coeffs
)

from ormatex_py.matexp_leja import (
    gen_leja_fast,
    gen_leja_conjugate,
    build_a_tilde,
    leja_shift_scale,
    real_leja_expmv_substep,
    complex_conj_leja_expmv_substep
)
try:
    import ormatex_py.ormatex as ormatex_rs
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False


class PhiEvaluatorModule(eqx.Module):

    @abstractmethod
    def eval_phi(self, k, cdt, b):
        raise NotImplementedError

    @abstractmethod
    def eval_phis(self, ks, cs, dt, bs):
        raise NotImplementedError


class PhiEvaluator():

    _valid_methods = {
        "krylov",
        "leja",
        "pfd",
        "pfd_rs",
        "dense"
    }

    def __init__(self, sys_lop, method="krylov", **kwargs):

        self.init_kwargs = kwargs

        # Phi evaluation method
        self.method = method
        if self.method not in self._valid_methods:
            raise AttributeError(f"{self.method} not in {self._valid_methods}")

        self.sys_lop = sys_lop
        if sys_lop is not None:
            self.set_lop(sys_lop)

    def set_lop(self, sys_lop):
        self.sys_lop = sys_lop

        if self.method == "krylov":
            self.Phi = PhiEvaluatorKrylov(sys_lop, **self.init_kwargs)
        elif self.method == "leja":
            self.Phi = PhiEvaluatorLeja(sys_lop, **self.init_kwargs)
        elif self.method == "pfd":
            self.Phi = PhiEvaluatorPFD(sys_lop, **self.init_kwargs)
        elif self.method == "pfd_rs":
            self.Phi = PhiEvaluatorPFDRS(sys_lop, **self.init_kwargs)
        elif self.method == "dense":
            self.Phi = PhiEvaluatorDense(sys_lop, **self.init_kwargs)
        else:
            raise NotImplementedError

        # TODO: fill out.
        # if self.method == "taylor":
        #    self.Phi = PhiEvaluatorTaylor(sys_lop, **self.init_kwargs)

    def getEvaluator(self) -> PhiEvaluatorModule:
        return self.Phi

    def eval_phi(self, k, cdt, b):
        r"""
        Evaluates a phi function application :math:`\varphi_k(\tau A) b` for k and b
        for the (intermediate) stepsize cdt (:math:`\tau = c*\Delta{t}`)
        """
        if not hasattr(self, 'Phi'):
            msg = "PhiEvaluator: must call set_lop or provide valid linear operator in init."
            raise ValueError(msg)
        return self.Phi.eval_phi(k, cdt, b)

    def eval_phis(self, ks, cdt, bs):
        r"""
        Evaluates a linear combination of phi function applications
        for all k in the tuple ks and vector rhs b in the tuple bs
        for the (intermediate) stepsize cdt (:math:`\tau = c*\Delta{t}`)
        .. math::
            w(\tau) = \sum_{j} \varphi_{k_j}(\tau A) b_{j}
        """
        if not hasattr(self, 'Phi'):
            msg = "PhiEvaluator: must call set_lop or provide valid linear operator in init."
            raise ValueError(msg)
        return self.Phi.eval_phis(ks, cdt, bs)


class PhiEvaluatorKrylov(PhiEvaluatorModule):
    sys_lop: LinOp
    max_krylov_dim: int
    iom: int

    _always_use_extension: bool
    _use_extension: bool

    def __init__(self, sys_lop, **kwargs):

        self.sys_lop = sys_lop

        # maximum krylov subspace dimension
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        # incomplete orthogonalization depth for mgs
        self.iom = kwargs.get("iom", 100)

        # configure the use of extension formulas
        self._always_use_extension = kwargs.get("always_use_extension", False)
        self._use_extension = kwargs.get("use_extension", True) or self._always_use_extension

    def eval_phi(self, k, cdt, b):
        if self._always_use_extension:
            return self.eval_phis((k,), cdt, (b,))
        else:
            result = phi_linop(self.sys_lop, cdt, b, k, self.max_krylov_dim, self.iom)
            return result

    def eval_phis(self, ks, cdt, bs):
        if self._use_extension:
            maxk = max(ks)
            rhs_list = [jnp.zeros(bs[0].shape)] * (maxk+1)
            for it, k in enumerate(ks):
                rhs_list[k] = rhs_list[k] + bs[it]
            # TODO: write new kiops which takes vb already as a block to avoid extra copies from list
            result = kiops_fixedsteps(
                self.sys_lop,
                cdt,
                rhs_list,
                max_krylov_dim=self.max_krylov_dim,
                iom=self.iom
            )
        else:
            result = jnp.zeros(bs[0].shape)
            for it, k in enumerate(ks):
                result += phi_linop(
                    self.sys_lop,
                    cdt,
                    bs[it],
                    k,
                    self.max_krylov_dim,
                    self.iom
                )
        return result


class PhiEvaluatorLeja(PhiEvaluatorModule):
    sys_lop: LinOp

    leja_tol: float
    leja_abc: tuple[float]

    leja_substep: bool
    leja_substep_size: int
    dd_method: str
    leja_n_zeros: int
    leja_max_power_iter: int
    leja_max_re_eig_scale: float
    n_leja: 280
    leja_x: jax.Array

    def __init__(self, sys_lop, **kwargs):
        # Relative tol for leja polynomial approx
        self.leja_tol = kwargs.get("leja_tol", 1e-15)
        # Option to enable substepping
        self.leja_substep = kwargs.get("leja_substep", True)
        # Initial substep size
        self.leja_substep_size = 1.0
        # Method used to compute divided diffs
        self.dd_method = kwargs.get("dd_method", "taylor")
        # number of repeated zeros prepended to the leja sequence
        self.leja_n_zeros = int(kwargs.get("leja_n_zeros", 2))
        # eigenvector corresponding to larget magnitude eigenvalue of sys jac.
        self.leja_max_power_iter = 100
        self.leja_max_re_eig_scale = kwargs.get("leja_max_re_eig_scale", 1.2)
        self.n_leja = kwargs.get("n_leja", 280)
        self.leja_x = jnp.asarray(gen_leja_fast(a=-2, b=2, n=self.n_leja))

        self.sys_lop = sys_lop

        # Optional max magnitude of real component of eigs(J*dt)
        leja_a = kwargs.get("leja_a", None)
        # Optional max magnitude of imag component of eigs(J*dt)
        leja_c = kwargs.get("leja_c", 0.)
        if leja_a is None:
            # estimate largest magnitude eigenvalue and corrosponding eigenvec
            # by power iter.
            _, _, leja_a, _, _power_iters = leja_shift_scale(
                sys_lop, sys_lop.n_domain, self.leja_max_power_iter,
                None, self.leja_max_re_eig_scale)
            print("=== Total power iters:  %d" % _power_iters)

        self.leja_abc = (leja_a, -leja_a, leja_c)

    def eval_phi(self, k, cdt, b):
        return self.eval_phis((k,), cdt, (b,))

    def eval_phis(self, ks, cdt, bs):
        maxk = max(ks)
        rhs_list = [jnp.zeros(bs[0].shape)] * (maxk+1)
        for it, k in enumerate(ks):
            rhs_list[k] = rhs_list[k] + bs[it]

        # build extended linop
        # TODO: write new method which takes vb already as a block to avoid extra copies from list
        a_tilde_lo, v, n = build_a_tilde(self.sys_lop, cdt, rhs_list)

        leja_a = self.leja_abc[0]
        shift = cdt*leja_a / 2.
        scale = np.abs(cdt*leja_a / 4.)

        # compute phi-vector products by leja interpolation
        result, leja_iters, converged, max_tau_dt = real_leja_expmv_substep(
            a_tilde_lo,
            1.1*self.leja_substep_size,
            v,
            self.leja_x,
            n,
            shift,
            scale,
            self.leja_tol,
            self.leja_substep,
            dd_method=self.dd_method)

        print("=== Total leja iters: %d, shift: %0.3f, scale: %0.3f" % (leja_iters, shift, scale))

        if not converged:
            raise RuntimeError("Leja not converged")

        return result

    """
    _power_iters = 0
        if self.leja_a is None:
            # estimate largest magnitude eigenvalue and corrosponding eigenvec
            # by power iter.  Store eigenvector for next step
            # to speed convergence of power iterations in
            # subsequent calls to power iter method.
            _, _, max_eig, self._leja_bk, _power_iters = leja_shift_scale(
                    a_tilde_lo, v.shape[0], self.leja_max_power_iter,
                    self._leja_bk, self.leja_max_re_eig_scale)
            leja_a = -jnp.abs(max_eig)
        else:
            leja_a = self.leja_a * dt
        if self.leja_c is None:
            leja_c = 0.0
        else:
            leja_c = self.leja_c * dt

        # generate leja sequence on the ellipse bounding the spectrum of the sys Jacobian
        leja_x, n_leja_real, scale, shift = gen_leja_conjugate(n=self.n_leja, a=leja_a, b=0., c=leja_c)
        leja_x = jnp.asarray(leja_x)

        # compute phi-vector products by leja interpolation
        y_update, leja_iters, converged, max_tau_dt = complex_conj_leja_expmv_substep(
                a_tilde_lo, 1.1*self.leja_substep_size, v, leja_x, n_leja_real,
                n, shift, scale, self.leja_tol, self.leja_substep,
                leja_n_zeros=self.leja_n_zeros, dd_method=self.dd_method)

        print("=t: %0.2f, Pwr itrs: %d, Leja itrs: %d, leja_a: %0.2f, leja_c: %0.2f, shift: %0.2f, scale: %0.2f" % (t, _power_iters, leja_iters, leja_a, leja_c, shift, scale))
        """


class PhiEvaluatorPFD(PhiEvaluatorModule):
    sys_lop: LinOp
    pfd_coeffs: tuple
    J: jax.Array

    def __init__(self, sys_lop, **kwargs):

        self.sys_lop = sys_lop

        # Partial fraction decomposition method
        pfd_method = kwargs.get("pfd_method", "cram_16")
        self.pfd_coeffs = get_pfd_coeffs(pfd_method)

        self.J = self.sys_lop.dense()

    @jax.jit
    def eval_phi(self, k, cdt, b):
        # cdtJ = cdt * self.sys_lop.dense()
        cdtJ = cdt * self.J
        return f_phi_ks_pfd(cdtJ, b, jnp.asarray(k), self.pfd_coeffs)

    @partial(jax.jit, static_argnames=('ks', ))
    def eval_phis(self, ks, cdt, bs):
        # cdtJ = cdt * self.sys_lop.dense()
        cdtJ = cdt * self.J

        Bs = jnp.stack(bs, axis=1)
        Ks = jnp.asarray(ks)
        results = f_phi_ks_pfd(cdtJ, Bs, Ks, self.pfd_coeffs)

        return jnp.sum(results, axis=1)


class PhiEvaluatorDense(PhiEvaluatorModule):
    sys_lop: LinOp
    J: jax.Array

    def __init__(self, sys_lop, **kwargs):
        self.sys_lop = sys_lop
        self.J = self.sys_lop.dense()

    @partial(jax.jit, static_argnames=('k', ))
    def eval_phi(self, k, cdt, b):
        return self.eval_phis((k,), cdt, (b,))

    @partial(jax.jit, static_argnames=('ks', ))
    def eval_phis(self, ks, cdt, bs):
        # cdtJ = cdt * self.sys_lop.dense()
        cdtJ = cdt * self.J

        phiJs = f_phi_k_sq_all(cdtJ, max(ks))

        result = jnp.zeros(bs[0].shape)
        for it, k in enumerate(ks):
            result += phiJs[k] @ bs[it]

        return result


class PhiEvaluatorPFDRS(PhiEvaluatorModule):
    sys_lop: LinOp
    phikv_dense_rs: eqx.Module
    J: np.array

    def __init__(self, sys_lop, **kwargs):

        self.sys_lop = sys_lop

        # Partial fraction decomposition method
        pfd_method = kwargs.get("pfd_method", "cram_16")
        pfd_order = kwargs.get("pfd_order", 16)

        if HAS_ORMATEX_RUST:
            self.phikv_dense_rs = ormatex_rs.DensePhikvEvalRs(pfd_method, pfd_order)
        else:
            raise AttributeError("PhiEvaluatorPFDRS requires the rust bindings, which were not found.")

        self.J = np.asarray(self.sys_lop.dense())

    def eval_phi(self, k, cdt, b):
        # J = np.asarray(self.sys_lop.dense())
        J = self.J
        result = self.phikv_dense_rs.eval(J, cdt, np.asarray(b).reshape(-1,1), k).flatten()
        return jnp.asarray(result)

    def eval_phis(self, ks, cdt, bs):
        # J = np.asarray(self.sys_lop.dense())
        J = self.J
        result = np.zeros(bs[0].shape)
        for it, k in enumerate(ks):
            result += self.phikv_dense_rs.eval(J, cdt, np.asarray(bs[it]).reshape(-1,1), k).flatten()
        return result
