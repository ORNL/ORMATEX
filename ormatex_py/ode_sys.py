##############################################################################
# Copyright© 2025 UT-Battelle, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""
Defines an interface for coupled systems of ODEs.
This interface is compatible with all time integration
methods in this ormatex_py package.
JAX allows user defined types as PyTrees:
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

    .. code-block::

        a_lo_aug =
            [[dt*A,  B],
             [0,     K]]

    where A is the original NxN linop
    dt is a scalar factor applied to A.
    b is a Nxp dense matrix.
    K is a pxp sparse matrix.
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
        # assert v.shape[0] == n + p
        ab_v = self.dt*self.a_lo(v[0:n]) + self.B @ v[-p:]
        k_v = self.K @ v[-p:]
        res = jnp.concat((ab_v, k_v))
        return res

    def _dense(self):
        raise NotImplementedError


class SysJacLinOp(LinOp):
    """
    Extend base LinOp class with pure virtual method for
    frhs time derivative and frhs caching.
    """
    _t: float
    _u: jax.Array
    _frhs: eqx.Module #type?
    _frhs_kwargs: dict

    def __init__(self, t, u, frhs: eqx.Module, frhs_kwargs: dict={}, **kwargs):
        self._t = t
        self._u = u
        # TODO: store the function handle with applied kwargs, however using Partial here leads to jax errors.
        # jax.tree_util.Partial(frhs, **frhs_kwargs)
        # There is no use of kwargs yet, decide how this is supposed to work.
        self._frhs = frhs
        self._frhs_kwargs = frhs_kwargs

    @abstractmethod
    def _fdt(self) -> jax.Array:
        """
        Prototype frhs time derivative
        """
        raise NotImplementedError

    def _frhs_cached(self) -> jax.Array:
        """
        Prototype cached frhs evaluation: evaluate frhs
        """
        return self._frhs(self._t, self._u, **self._frhs_kwargs)


class CustomJacLinOp(SysJacLinOp):
    _f_du: jax.Array
    _f_dt: jax.Array

    # for finite difference fallback
    _frhs_u: jax.Array
    _eps: float

    def __init__(self, t, u, frhs: eqx.Module, f_du: jax.Array, f_dt: jax.Array=None, frhs_kwargs: dict={}, **kwargs):
        super().__init__(t, u, frhs, frhs_kwargs, **kwargs)
        self._f_du = MatrixLinOp(f_du)
        self._f_dt = f_dt

        if self._f_dt is None:
            self._frhs_u = self._frhs(t, u, **self._frhs_kwargs)
            self._eps = kwargs.get("eps", 0.5e-8)
        else:
            self._frhs_u = None
            self._eps = None

    @jax.jit
    def _fdt(self) -> jax.Array:
        if self._f_dt is None:
            # if f_dt is not supplied, use finite difference fallback
            eps = self._eps
            f_dt = (self._frhs(self._t + eps, self._u, **self._frhs_kwargs) - self._frhs_u) / eps
            return f_dt
        else:
            return self._f_dt

    @jax.jit
    def _matvec(self, b: jax.Array) -> jax.Array:
        # delegate _matvec to owned MatrixLinOp
        return self._f_du._matvec(b)

    @jax.jit
    def _dense(self) -> jax.Array:
        # delegate _dense to owned MatrixLinOp
        return self._f_du._dense()

    @jax.jit
    def _frhs_cached(self) -> jax.Array:
        """
        Prototype cached frhs evaluation: evaluate frhs
        """
        if self._frhs_u is None:
            return self._frhs(self._t, self._u, **self._frhs_kwargs)
        else:
            # for finite difference fallback
            return self._frhs_u


class AdJacLinOp(SysJacLinOp):
    _frhs_u: jax.Array
    _fjac_u: Callable

    def __init__(self, t, u, frhs: eqx.Module, frhs_kwargs: dict={}, **kwargs):
        super().__init__(t, u, frhs, frhs_kwargs, **kwargs)
        self._frhs_u, self._fjac_u = jax.linearize(partial(self._frhs, **self._frhs_kwargs), self._t, self._u)

    @jax.jit
    def _matvec(self, v: jax.Array) -> jax.Array:
        """
        Define the action of the Jacobian of frhs on a vector v.

        Args:
            v: target vector to apply linop to
        """
        print("jit-compiling AdJacLinOp._matvec")
        return self._fjac_u(0., v.reshape(self._u.shape))

    @jax.jit
    def _fdt(self) -> jax.Array:
        """
        frhs time derivative
        """
        print("jit-compiling AdJacLinOp._fdt")
        return self._fjac_u(1., jnp.zeros(self._u.shape))

    @jax.jit
    def _frhs_cached(self) -> jax.Array:
        """
        cached frhs evaluation
        """
        return self._frhs_u

    @jax.jit
    def _dense(self):
        """
        Define the (dense) jacobian of frhs.
        """
        return jax.vmap(self._matvec, in_axes=(1), out_axes=1)(jnp.eye(self._u.shape[0]))


class FdJacLinOp(SysJacLinOp):
    # storage for rhs evaluated at u, saves computation
    # on repeated calls to jvp
    _frhs_u: jax.Array
    _scale: float
    _gamma: float
    _eps: float

    def __init__(self, t, u, frhs: Callable, frhs_kwargs: dict={},
                 scale: float=1.0, gamma: float=0.0, **kwargs):
        super().__init__(t, u, frhs, frhs_kwargs, **kwargs)
        self._frhs_u = self._frhs(t, u, **self._frhs_kwargs)
        self._scale = scale
        self._gamma = gamma # shift
        self._eps = kwargs.get("eps", 0.5e-8)

    @property
    def shape(self) -> (int, int):
        """
        LinOp shape
        """
        return (self._u.size, self._u.size)

    @jax.jit
    def _matvec(self, v: jax.Array) -> jax.Array:
        """
        Define action of linop on a vector.
        Args:
            v: target vector to apply linop to
        """
        u_norm_1 = jnp.linalg.norm(self._u, 1)
        scaled_eps = self._eps * u_norm_1
        ieps = self._scale / scaled_eps
        u_pert = self._u + scaled_eps*v

        # compute the unshifted jac-vec product.
        diff = self._frhs(self._t, u_pert, **self._frhs_kwargs) - self._frhs_u
        j_v = (diff)*ieps

        # shift j_v product
        j_v += self._gamma * v
        return j_v

    @jax.jit
    def _fdt(self) -> jax.Array:
        """
        Computes time derivative of the rhs by finite difference
        """
        print("jit-compiling FdJacLinOp._fdt")
        return (self._frhs(self._t+self._eps, self._u, **self._frhs_kwargs) - self._frhs_u) / self._eps

    @jax.jit
    def _frhs_cached(self) -> jax.Array:
        """
        cached frhs evaluation
        """
        return self._frhs_u

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
        return AdJacLinOp(t, u, self.frhs, frhs_kwargs=kwargs)

    def _fm(self, t: float, u: jax.Array, **kwargs) -> LinOp:
        # default implementation (identity operator)
        return EyeLinOp(u.shape[0])


class OdeSplitSys(OdeSys):
    """
    Define a split ODE system.

    .. code-block::

      dU/dt = F(t, U) = L(t, U) @ U + R(t, U),

    where L(t, U) is a (potentially time and state dependent) LinOp,
    different from the Jacobian.
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

        .. code-block::

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
        inner_frhs = self.inner._frhs(t, x.flatten(), **kwargs).reshape((-1, ), order='F')
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
