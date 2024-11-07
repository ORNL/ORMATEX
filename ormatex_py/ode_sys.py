"""
Defines an interface for coupled systems of ODEs.
This interface is compatible with all time integration
methods in this ormatex_py package.

TODO: Needs updates for jax compatibility.
jax allows user defined types as PyTrees:
See: https://jax.readthedocs.io/en/latest/pytrees.html

NOTE: this equinox package may make jax compatibility
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

    @abstractmethod
    def _dense(self):
        raise NotImplementedError

    def dense(self) -> jax.Array:
        return self._dense()


class JaxMatrixLinop(LinOp):
    """
    Helper class to wrap a jax.Array as a LinOp
    similar to scipy.sparse.linalg.aslinearoperator()
    """
    a: jax.Array | jsp.JAXSparse
    a_dense: jax.Array

    def __init__(self, a):
        assert len(a.shape) == 2
        assert a.shape[0] == a.shape[1]
        self.a = a
        self.a_dense = None

    def _matvec(self, b: jax.Array) -> jax.Array:
        return self.a @ b

    def _dense(self) -> jax.Array:
        ## TODO: I do not know why this class is frozen, and what the consequences of super().__setattr__ are
        # without this, I get: FrozenInstanceError: cannot assign to field 'a_dense'
        if self.a_dense is None:
            try:
                super().__setattr__('a_dense', self.a.todense())
            except AttributeError:
                super().__setattr__('a_dense', self.a)
        return self.a_dense


class AugMatrixLinop(LinOp):
    """
    Helper class to create a larger linear operator
    out of a block matrix comprised of components:
    a_lo_aug =
        [[A,  B],
         [0,  K]]
    where A is the original NxN linop
    b is a Nxp dense matrix
    K is a pxp sparse matrix
    """
    a_lo: Callable
    # scalar factor applied to a_lo
    dt: float
    B: jax.Array
    K: jax.Array
    n: int
    p: int

    def __init__(self, a_lo, dt, B, K):
        self.a_lo = a_lo
        self.dt = dt
        self.B = B
        self.K = K
        self.n = self.B.shape[0]
        self.p = self.B.shape[1]
        assert self.p == self.K.shape[0]
        assert self.p == self.K.shape[1]

    def _matvec(self, v):
        """
        Computes a_lo_aug @ v
        """
        assert v.shape[0] == self.n + self.p
        ab_v = self.dt*self.a_lo(v[0:self.n]) + self.B @ v[-self.p:]
        k_v = self.K @ v[-self.p:]
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

    def _matvec(self, v: jax.Array) -> jax.Array:
        """
        Define the action of the jacobian of frhs on a vector v.

        Args:
            v: target vector to apply linop to
        """
        return self.fjac_u(v)

    def _dense(self):
        """
        Define the (dense) jacobian of frhs.
        """
        return jax.vmap(self.fjac_u, in_axes=(1), out_axes=1)(jnp.eye(self.u.shape[0]))


## TODO: this class is only needed for testing going forward?
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
                 scale: float=1.0,  gamma: float=0.0, *args, **kwargs):
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
        Linop shape
        """
        return (self.u.size, self.u.size)

    def set_op_u(self, t: float, u: jax.Array):
        """
        Set state at which to linearize the system
        TODO: not used, do we need it?
        """
        self.t = t
        self.u = u
        self.frhs_u = self.frhs(t, u, **self.frhs_kwargs)

    def set_scale(self, scale: float, gamma: float):
        """
        For optional scaling and shifting of the Jv product
        TODO: scale, gamma and this method are not used, do we need it?
        """
        self.scale = scale
        self.gamma = gamma

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

        # compute the ushifited jac-vec product.
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
        where M=const, A=A(U,t), B=B(U,t) are matricies
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
        dF(t, U)/dU|_(t,u) = M^-1 d[AU + BU + S]/dU |_(t,u)
        where M, A, B are matricies and S is a vec. and M is const.

        The default implementation uses a finite diff approx
        to construct this linear operator, but is highly inefficient.
        NOTE: Recommend to override this method.

        Args:
            t:  current time
            u:  current system state.
        """
        return self._fjac(t, u, **kwargs)

    def fm(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        """
        Method that returns the mass matrix lin op
        """
        return self._fm(t, u, **kwargs)

    @abstractmethod
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        # the user must override this
        raise NotImplementedError

    #@partial(jax.jit(static_argnums=(0,)))
    def _fjac(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        # default implementation
        #return FdJacLinOp(t, u, self.frhs, frhs_kwargs=kwargs)
        return JacLinOp(t, u, self.frhs, frhs_kwargs=kwargs)

    def _fm(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        # default implementation
        return lambda x: x


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
        self.t = t0
        self.order = order
        self.method = method
        self.t_hist = deque(maxlen=order)
        self.y_hist = deque(maxlen=order)
        self.t_hist.appendleft(t0)
        self.y_hist.appendleft(y0)

    def reset_ic(self, t0: float, y0: jax.Array):
        self.t = t0
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
