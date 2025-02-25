"""
Defines an interface for coupled systems of ODEs.
This interface is compatible with all time integration
methods in this ormatex_py package.

jax allows user defined types as PyTrees:
See: https://jax.readthedocs.io/en/latest/pytrees.html
NOTE: we use equinox to make jax compatibility
simpler by removing boilerplate to register objects as pytrees.
from discussion here:  https://github.com/jax-ml/jax/discussions/10598
"""
from abc import ABCMeta
from abc import abstractmethod, abstractproperty
from collections.abc import Callable
import numpy as np
from functools import partial
from collections import deque
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
import equinox as eqx
from dataclasses import dataclass


class LinOp(eqx.Module):

    @abstractmethod
    def _matvec(self, v):
        raise NotImplementedError

    def __call__(self, v):
        return self._matvec(v)

    def matvec(self, v: jax.Array) -> jax.Array:
        return self._matvec(v)

    def matvec_npcompat(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(self._matvec(v)).flatten()

    @abstractmethod
    def _dense(self):
        raise NotImplementedError

    def dense(self) -> jax.Array:
        return self._dense()


class EyeLinOp(LinOp):
    """
    Identity LinOp
    """

    n: int

    def __init__(self, n: int):
        self.n = n

    @jax.jit
    def _matvec(self, v):
        return v

    def _dense(self):
        return jnp.eye(self.n)


class DiagLinOp(LinOp):
    """
    Diagonal linear operator with diagonal d
    """

    d: jax.Array

    def __init__(self, d: jax.Array):
        self.d = d

    @jax.jit
    def _matvec(self, v):
        return self.d * v

    def _dense(self):
        return jnp.diag(self.d)


class MatrixLinOp(LinOp):
    """
    Helper class to wrap a jax.Array as a LinOp
    similar to scipy.sparse.linalg.aslinearoperator()
    """
    a: jax.Array | jsp.JAXSparse

    def __init__(self, a):
        assert len(a.shape) == 2
        assert a.shape[0] == a.shape[1]
        self.a = a

    @jax.jit
    def _matvec(self, b: jax.Array) -> jax.Array:
        print("jit-compiling MatrixLinOp._matvec")
        return self.a @ b

    def _dense(self) -> jax.Array:
        if isinstance(self.a, jax.Array):
            return self.a
        else:
            #convert sparse array to dense
            return self.a.todense()


class AugMatrixLinOp(LinOp):
    """
    Helper class to create a larger linear operator
    out of a block matrix comprised of components:
    a_lo_aug =
        [[dt*A,  B],
         [0,     K]]
    where A is the original NxN linop
    dt is a scalar factor applied to A
    b is a Nxp dense matrix
    K is a pxp sparse matrix
    """
    a_lo: LinOp
    dt: float
    B: jax.Array
    K: jax.Array

    def __init__(self, a_lo, dt, B, K):
        self.a_lo = a_lo
        self.dt = dt
        self.B = B
        self.K = K
        assert self.B.shape[1] == self.K.shape[0]
        assert self.B.shape[1] == self.K.shape[1]

    @jax.jit
    def _matvec(self, v):
        """
        Computes a_lo_aug @ v
        """
        print("jit-compiling AugMatrixLinOp._matvec")
        n = self.B.shape[0]
        p = self.B.shape[1]
        assert v.shape[0] == n + p
        ab_v = self.dt*self.a_lo(v[0:n]) + self.B @ v[-p:]
        k_v = self.K @ v[-p:]
        res = jnp.concat((ab_v, k_v))
        return res

    def _dense(self):
        raise NotImplementedError


class JacLinOp(LinOp):
    t: float
    u: jax.Array
    frhs: eqx.Module
    # storage for rhs evaluated at u, computed in linerize call
    # TODO: figure out how to eliminate this redundant computation
    frhs_u: jax.Array
    frhs_kwargs: dict
    fjac_u: Callable

    def __init__(self, t, u, frhs: eqx.Module, frhs_kwargs: dict={}, **kwargs):
        self.t = t
        self.u = u
        self.frhs = frhs
        self.frhs_kwargs = frhs_kwargs
        self.frhs_u, self.fjac_u = jax.linearize(partial(frhs, self.t, **self.frhs_kwargs), self.u)

    @jax.jit
    def _matvec(self, v: jax.Array) -> jax.Array:
        """
        Define the action of the Jacobian of frhs on a vector v.

        Args:
            v: target vector to apply linop to
        """
        print("jit-compiling JacLinOp._matvec")
        return self.fjac_u(v.reshape(self.u.shape))

    def _dense(self):
        """
        Define the (dense) jacobian of frhs.
        """
        return jax.vmap(self.fjac_u, in_axes=(1), out_axes=1)(jnp.eye(self.u.shape[0]))


class FdJacLinOp(LinOp):
    t: float
    u: jax.Array
    frhs: Callable
    # storage for rhs evaluated at u, saves computation
    # on repeated calls to jvp
    frhs_u: jax.Array
    frhs_kwargs: dict
    scale: float
    gamma: float
    eps: float

    def __init__(self, t, u, frhs: Callable, frhs_kwargs: dict={},
                 scale: float=1.0, gamma: float=0.0, **kwargs):
        self.t = t
        self.u = u
        self.frhs = frhs
        self.frhs_u = self.frhs(t, u, **frhs_kwargs)
        self.frhs_kwargs = frhs_kwargs
        # scale
        self.scale = scale
        # shift
        self.gamma = gamma
        self.eps = kwargs.get("eps", 0.5e-8)

    @property
    def shape(self) -> (int, int):
        """
        LinOp shape
        """
        return (self.u.size, self.u.size)

    @jax.jit
    def _matvec(self, v: jax.Array) -> jax.Array:
        """
        Define action of linop on a vector.
        Args:
            v: target vector to apply linop to
        """
        u_norm_1 = jnp.linalg.norm(self.u, 1)
        scaled_eps = self.eps * u_norm_1
        ieps = self.scale / scaled_eps
        u_pert = self.u + scaled_eps*v

        # compute the unshifted jac-vec product.
        diff = self.frhs(self.t, u_pert, **self.frhs_kwargs) - self.frhs_u
        j_v = (diff)*ieps

        # shift j_v product
        j_v += self.gamma * v
        return j_v

    def _dense(self):
        raise NotImplementedError


# NOTE:  https://docs.kidger.site/equinox/api/module/module/
# every eqx.Module is an abstract base class by default
class OdeSys(eqx.Module):
    def __init__(self, *args, **kwargs):
        pass

    def frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        """
        Define the RHS of the system.
        dU/dt = F(t, U)

        Ex: Could look like:
        F(t, U) = M^-1 [AU + BU + S]
        where M=const, A, B are matricies
        and S=S(u,t) is a vec.

        Args:
            t:  current time
            u:  current system state.
        """
        return self._frhs(t, u, **kwargs)

    def __call__(self, t: float, u: jax.Array, *args, **kwargs) -> jax.Array:
        """
        Alias to _frhs for diffrax compatibility.
        """
        return self._frhs(t, u, **kwargs)

    def fjac(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        """
        Define the Jacobian of the system as a linear operator

        Ex: Could look like:
        dF(t, U)/dU|_(t,u) = M^-1 (A + B + dS/dU|_(t,u))
        where M, A, B are matricies and S is a vec. and M is const.

        The default implementation uses jax.linearize.

        Args:
            t:  current time
            u:  current system state.
        """
        return self._fjac(t, u, **kwargs)

    def fm(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        """
        Method that returns the mass matrix LinOp
        """
        return self._fm(t, u, **kwargs)

    @abstractmethod
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        # the user must override this
        raise NotImplementedError

    @jax.jit
    def _fjac(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        # default implementation (autodiff Jacobian)
        print("jit-compiling ODESys._fjac")
        return JacLinOp(t, u, self.frhs, frhs_kwargs=kwargs)

    def _fm(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        # default implementation (identity operator)
        return EyeLinOp(u.shape[0])


class OdeSplitSys(OdeSys):
    """
    Define a split ODE system.
      dU/dt = F(t, U) = L(t, U) @ U + R(t, U),
    where L(t, U) is a (potentially time and state dependent) LinOp, different from the Jacobian.
    """
    def __init__(self, *args, **kwargs):
        pass

    def fl(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        """
        Define the LinOp L(t,u) of the system.
        Args:
            t:  current time
            u:  current system state.
        """
        return self._fl(t, u, **kwargs)

    @abstractmethod
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        """
        Define the RHS of the split system
          dU/dt = F(t, U) = L(t, U) @ U + R(t, U).

        Since we need residual quantities
          R_0 = F(t, U) - F(t_0, U_0) - L(t_0,U_0) @ (U - U0)
        which is not the same as R(t, U) - R(t_0, U_0), it is easier to
        specify F(t, U) and L(t, U), which implicitly defines
          R(t, U) = F(t, U) - L(t, U) @ U,
        which however is not explicitly needed by the integrators.

        Args:
            t:  current time
            u:  current system state.
        """
        # the user must override this
        raise NotImplementedError

    @abstractmethod
    def _fl(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        # the user must override this
        raise NotImplementedError


class OdeSysNp(OdeSys):
    inner: OdeSys

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def _frhs(self, t, x, **kwargs):
        # Convert jax array to numpy array for Rust compat
        # Alternatively, a pure numpy implementation is acceptable,
        # but JAX can be powerful for automatic differentiation
        # and GPU acceleration of the system model.
        inner_frhs = self.inner._frhs(t, x, **kwargs).reshape((-1, ), order='F')
        return np.asarray(inner_frhs)

    def _fjac(self, t, x, **kwargs):
        return self.inner._fjac(t, x.flatten(), **kwargs)


@dataclass
class StepResult:
    # time at end of the step
    t: float
    # step size taken
    dt: float
    # state after step
    y: jax.Array
    # step error estimate
    err: float


class IntegrateSys(metaclass=ABCMeta):
    """
    Defines interface to time integrators
    """

    # list of valid methods
    _valid_methods = {}

    # defines the rhs of the system of odes
    sys: OdeSys
    # current time
    t: float
    # additional info and storage
    order: int
    method: str
    t_hist: deque[float]
    y_hist: deque[jax.Array]

    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, order: int, method: str, *args, **kwargs):
        assert order > 0
        self.sys = sys
        self.t = float(t0)
        self.order = order
        self.method = method
        self.t_hist = deque(maxlen=order)
        self.y_hist = deque(maxlen=order)
        self.t_hist.appendleft(t0)
        self.y_hist.appendleft(y0)

    def reset_ic(self, t0: float, y0: jax.Array):
        self.t = float(t0)
        self.t_hist.clear()
        self.y_hist.clear()
        self.t_hist.appendleft(t0)
        self.y_hist.appendleft(y0)

    @abstractmethod
    def step(self, dt: float) -> StepResult:
        raise NotImplementedError

    def accept_step(self, s: StepResult):
        """
        default implementation, maybe overridden
        """
        self.t = s.t
        self.t_hist.appendleft(s.t)
        self.y_hist.appendleft(s.y)

    @property
    def time(self) -> float:
        """
        current time
        """
        return self.t

    @property
    def state(self) -> jax.Array:
        """
        current state
        """
        self.y_hist[0]
