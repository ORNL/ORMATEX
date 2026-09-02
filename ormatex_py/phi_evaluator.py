"""
Exponential time integration methods
"""
import jax
import jax.numpy as jnp
import numpy as np

from abc import abstractmethod
from typing import Optional
import equinox as eqx

from functools import partial

from ormatex_py.ode_sys import LinOp

from ormatex_py.matexp_krylov import phi_linop, kiops_fixedsteps
from ormatex_py.matexp_phi import (
    f_phi_k_sq_all,
    PhiEvaluator_PFD_Dense,
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


class PhiPlan(eqx.Module):
    r"""
    Provides a plan of
    - timestep size dt (converted to a jax.Array)
    - local timestep sizes cdt = c*dt (converted to a jax.Array,
      to ensure we do not introduce any new dtype conversions)
    - and phi orders k as a list of lists of int
    to PhiEvaluatorModule at init time.
    """
    dt: jax.Array = eqx.field(converter=jnp.asarray)
    cdts: jax.Array = eqx.field(converter=jnp.asarray)
    ks: tuple[tuple[int]] = eqx.field(static=True)

    def __check_init__(self):
        assert(len(self.dt.shape) == 0)
        assert(len(self.cdts.shape) == 1)
        assert(len(self.cdts) == len(self.ks))
        assert(all(isinstance(k, tuple) for k in self.ks))


class PhiEvaluatorModule(eqx.Module):
    r"""
    Abstract equinox module for PhiEvaluator implementations
    """
    sys_lop: LinOp
    phi_plan: PhiPlan

    def __init__(self, sys_lop: LinOp, phi_plan: Optional[PhiPlan] = None):
        self.sys_lop = sys_lop
        self.phi_plan = phi_plan if phi_plan is not None else PhiPlan(0., tuple(), tuple())

    @abstractmethod
    def eval_phi(self, k: int, cdt: float, b: jax.Array):
        raise NotImplementedError

    @abstractmethod
    def eval_phis(self, ks: tuple[int], cdt: float, bs:  tuple[jax.Array]):
        raise NotImplementedError

    def eval_phis_plan(self, bs: jax.Array, step: int):
        ks = self.phi_plan.ks[step]
        cdt = self.phi_plan.cdts[step]
        return self.eval_phis(ks, cdt, bs)


class PhiEvaluator():

    _valid_methods = {
        "krylov",
        "leja",
        "pfd",
        "pfd_rs",
        "dense"
    }

    def __init__(self,
                 method: str = "krylov",
                 sys_lop: Optional[LinOp] = None,
                 phi_plan: Optional[PhiPlan] = None,
                 **kwargs):

        self.init_kwargs = kwargs

        # Phi evaluation method
        self.method = method
        if self.method not in self._valid_methods:
            raise AttributeError(f"{self.method} not in {self._valid_methods}")

        self.sys_lop = sys_lop
        if sys_lop is not None:
            self.set_lop(sys_lop, phi_plan=phi_plan)

    def set_lop(self,
                sys_lop: LinOp,
                phi_plan: Optional[PhiPlan] = None,):
        self.sys_lop = sys_lop

        if self.method == "krylov":
            self.Phi = PhiEvaluatorKrylov(sys_lop, phi_plan=phi_plan, **self.init_kwargs)
        elif self.method == "leja":
            self.Phi = PhiEvaluatorLeja(sys_lop, phi_plan=phi_plan, **self.init_kwargs)
        elif self.method == "pfd":
            self.Phi = PhiEvaluatorPFD(sys_lop, phi_plan=phi_plan, **self.init_kwargs)
        elif self.method == "pfd_rs":
            self.Phi = PhiEvaluatorPFDRS(sys_lop, phi_plan=phi_plan, **self.init_kwargs)
        elif self.method == "dense":
            self.Phi = PhiEvaluatorDense(sys_lop, phi_plan=phi_plan, **self.init_kwargs)
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

    def eval_phis_plan(self, bs, step):
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
        return self.Phi.eval_phis_plan(bs, step)


class PhiEvaluatorKrylov(PhiEvaluatorModule):
    max_krylov_dim: int
    iom: int

    _always_use_extension: bool = eqx.field(static=True)
    _use_extension: bool = eqx.field(static=True)

    def __init__(self, sys_lop, phi_plan=None, **kwargs):

        super().__init__(sys_lop, phi_plan)

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

    def __init__(self, sys_lop, phi_plan=None, **kwargs):

        super().__init__(sys_lop, phi_plan)

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
    J: jax.Array
    pfd_method: str = eqx.field(static=True)

    pfd_plan_list: list[PhiEvaluator_PFD_Dense]

    def __init__(self, sys_lop, phi_plan=None, **kwargs):

        super().__init__(sys_lop, phi_plan)

        # Partial fraction decomposition method
        self.pfd_method = kwargs.get("pfd_method", "cram_16")

        self.J = self.sys_lop.dense()

        self.pfd_plan_list = [None] * len(self.phi_plan.cdts)
        if len(self.pfd_plan_list) > 0:
            # unique logic is hard to jit-compile, use numpy unique
            # since it is faster than jax.
            # TODO: ideally this computation should be done statically
            # for each integrator plan once, not here in every step
            cdtu, inverse = np.unique(self.phi_plan.cdts, return_inverse=True)
            # precompute all lu-decompositions that will be needed
            pfd_list = [PhiEvaluator_PFD_Dense(float(cdt) * self.J, self.pfd_method) for cdt in cdtu]
            self.pfd_plan_list = [pfd_list[ind] for ind in inverse]

    @jax.jit
    def eval_phi(self, k, cdt, b):
        cdtJ = cdt * self.J
        phi_pfd = PhiEvaluator_PFD_Dense(cdtJ, self.pfd_method)
        return phi_pfd.apply(b, jnp.asarray(k))

    @partial(jax.jit, static_argnames=('ks', ))
    def eval_phis(self, ks, cdt, bs):
        cdtJ = cdt * self.J
        bs = jnp.stack(bs, axis=1)
        ks = jnp.asarray(ks)
        phi_pfd = PhiEvaluator_PFD_Dense(cdtJ, self.pfd_method)
        results = phi_pfd.apply(bs, ks)
        return jnp.sum(results, axis=1)

    @jax.jit(static_argnums=(2,))
    def eval_phis_plan(self, bs, step: int):
        bs = jnp.stack(bs, axis=1)
        ks = jnp.asarray(self.phi_plan.ks[step])
        phi_pfd = self.pfd_plan_list[step]
        results = phi_pfd.apply(bs, ks)
        return jnp.sum(results, axis=1)


class PhiEvaluatorDense(PhiEvaluatorModule):
    J: jax.Array

    def __init__(self, sys_lop, phi_plan=None, **kwargs):
        super().__init__(sys_lop, phi_plan)
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

    @jax.jit(static_argnums=(2,))
    def eval_phis_plan(self, bs: jax.Array, step: int):
        ks = self.phi_plan.ks[step]
        cdt = self.phi_plan.cdts[step]
        return self.eval_phis(ks, cdt, bs)


class PhiEvaluatorPFDRS(PhiEvaluatorModule):
    phikv_dense_rs: eqx.Module
    J: np.array

    def __init__(self, sys_lop, phi_plan=None, **kwargs):

        super().__init__(sys_lop, phi_plan)

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
