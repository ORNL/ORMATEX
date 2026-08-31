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
from ormatex_py.matexp_phi import f_phi_k_sq_all, f_phi_ks_pfd, get_pfd_coeffs
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

    _valid_methods = {"krylov", "kiops", "pfd", "pfd_rs", "dense"}

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
        elif self.method == "kiops":
            self.Phi = PhiEvaluatorKIOPS(sys_lop, **self.init_kwargs)
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
        # if self.method == "leja":
        #    self.Phi = PhiEvaluatorLeja(sys_lop, **self.init_kwargs)

    def getEvaluator(self) -> PhiEvaluatorModule:
        return self.Phi

    def eval_phi(self, k, cdt, b):
        r"""
        Evaluates a phi function application :math:`\varphi_k(\tau A) b` for k and b
        for the (intermediate) stepsize cdt (:math:`\tau = c*\Delta{t}`)
        """
        if not isinstance(self.Phi, PhiEvaluatorModule):
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
        if not isinstance(self.Phi, PhiEvaluatorModule):
            msg = "PhiEvaluator: must call set_lop or provide valid linear operator in init."
            raise ValueError(msg)
        return self.Phi.eval_phis(ks, cdt, bs)


class PhiEvaluatorKrylov(PhiEvaluatorModule):
    sys_lop: LinOp
    max_krylov_dim: int
    iom: int

    def __init__(self, sys_lop, **kwargs):

        self.sys_lop = sys_lop

        # maximum krylov subspace dimension
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        # incomplete orthogonalization depth for mgs
        self.iom = kwargs.get("iom", 100)

    def eval_phi(self, k, cdt, b):
        result = phi_linop(self.sys_lop, cdt, b, k, self.max_krylov_dim, self.iom)
        return result

    def eval_phis(self, ks, cdt, bs):
        result = jnp.zeros(bs[0].shape)
        for it, k in enumerate(ks):
            result += phi_linop(self.sys_lop, cdt, bs[it], k, self.max_krylov_dim, self.iom)
        return result


class PhiEvaluatorKIOPS(PhiEvaluatorModule):
    sys_lop: LinOp
    max_krylov_dim: int
    iom: int

    def __init__(self, sys_lop, **kwargs):

        self.sys_lop = sys_lop

        # maximum krylov subspace dimension
        self.max_krylov_dim = kwargs.get("max_krylov_dim", 100)
        # incomplete orthogonalization depth for mgs
        self.iom = kwargs.get("iom", 100)

    def eval_phi(self, k, cdt, b):
        return self.eval_phis((k,), cdt, (b,))

    def eval_phis(self, ks, cdt, bs):
        maxk = max(ks)
        rhs_list = [jnp.zeros(bs[0].shape)] * (maxk+1)
        for it, k in enumerate(ks):
            rhs_list[k] = rhs_list[k] + bs[it]
        # TODO: write new kiops which takes vb already as a block to avoid extra copies from list
        result = kiops_fixedsteps(
            self.sys_lop, cdt, rhs_list,
            max_krylov_dim=self.max_krylov_dim, iom=self.iom)
        return result


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
