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
from typing import Callable, List, Deque
import numpy as np
from scipy.sparse.linalg import LinearOperator
from functools import partial
from collections import deque
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass


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


class FdJacLinOp(eqx.Module, LinearOperator):
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

    def __init__(self, t, u, frhs: Callable, frhs_kwargs: dict,
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
        """
        self.t = t
        self.u = u
        self.frhs_u = self.frhs(t, u, **self.frhs_kwargs)

    def set_scale(self, scale: float, gamma: float):
        """
        For optional scaling and shifting of the Jv product
        """
        self.scale = scale
        self.gamma = gamma

    def __call__(self, v: jax.Array) -> jax.Array:
        """
        Alias to linop-vector product
        """
        return self._matvec(v)

    def _matvec(self, v: jax.Array) -> jax.Array:
        """
        Define action of linop on a vector.
        NOTE:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
        This defines the @ operator for this class

        Args:
            v: target vector to apply linop to
        """
        u_norm_1 = jnp.linalg.norm(self.u, 1)
        scaled_eps = self.eps * u_norm_1
        ieps = self.scale / scaled_eps
        u_pert = self.u + scaled_eps*v

        # compute the ushifited jac-vec product.
        j_v = (self.frhs(self.t, u_pert, **self.frhs_kwargs) - self.frhs_u)*ieps

        # shift j_v product
        j_v += self.gamma * v

        return j_v


class OdeSys(metaclass=ABCMeta):
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

    def fjac(self, t: float, u: jax.Array, **kwargs) -> LinearOperator:
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

    @abstractmethod
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        # the user must ovrride this
        raise NotImplementedError

    def frhs_aug(self, t: float, u: jax.Array, aug: jax.Array, aug_scale: float, **kwargs) -> jax.Array:
        return aug_scale * self.frhs(t, u, **kwargs) + aug

    # @partial(jax.jit(static_argnums=(0,)))
    def _fjac(self, t: float, u: jax.Array, **kwargs) -> LinearOperator:
        # default implementation
        return FdJacLinOp(t, u, self.frhs, frhs_kwargs=kwargs)


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
    t_hist: Deque[float]
    y_hist: Deque[jax.Array]

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
