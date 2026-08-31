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
import jax
import jax.numpy as jnp
from functools import partial
from ormatex_py.ode_sys import LinOp, IntegrateSys, OdeSys, OdeSplitSys, StepResult


@IntegrateSys.populate_valid_methods
class RKIntegrator(IntegrateSys):

    # new class list of valid methods
    _valid_methods = {}

    def __init__(self, sys: OdeSys, t0: float, y0: jax.Array, method="rk4", **kwargs):
        self.method = method
        if self.method not in self._valid_methods.keys():
            raise AttributeError(f"{self.method} not in {self._valid_methods}")
        order = self._valid_methods[self.method].order
        super().__init__(sys, t0, y0, order, self.method, **kwargs)

    @jax.jit
    def _step_rk4_jit(t, yt, dt, sys):
        print("jit-compiling rk4 kernel")

        f1 = sys.frhs(t,      yt)
        f2 = sys.frhs(t+dt/2, yt + f1*dt/2)
        f3 = sys.frhs(t+dt/2, yt + f2*dt/2)
        f4 = sys.frhs(t+dt,   yt + f3*dt)

        y_new = yt + dt * (f1 + 2*f2 + 2*f3 + f4) / 6;

        # no error est. avail
        y_err = -1.

        return y_new, y_err

    @IntegrateSys.register_method(["rk4"], 4)
    def _step_rk4(self, dt: float) -> StepResult:
        """
        Computes the solution update by RK4
        """
        t = self.t
        yt = self.y_hist[0]

        y_new, y_err = RKIntegrator._step_rk4_jit(t, yt, dt, self.sys)

        return StepResult(t+dt, dt, y_new, y_err)

    def step(self, dt: float) -> StepResult:
        return self._valid_methods[self.method](self, dt)
