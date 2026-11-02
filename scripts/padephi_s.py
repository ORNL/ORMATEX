##############################################################################
# Copyright© 2026 UT-Battelle, LLC
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
from mpmath import mp


def padephi_s(
        m: int,
        n: int,
        s: int = 0,
    ):

    # only implemented for s==0 and (sub-)diagonal
    assert(s == 0)
    assert(m <= n)

    # Set precision
    mp.dps = 40

    # pade approx of exponential
    m_range = list(range(m + 1))
    n_range = list(range(n + 1))
    a = [mp.binomial(m, mi) / mp.binomial(n+m, mi) / mp.factorial(mi) for mi in m_range]
    b = [mp.binomial(n, ni) / mp.binomial(n+m, ni) / mp.factorial(ni) * (-1)**(ni) for ni in n_range]

    # Normalize.
    # a = a/b(1);
    # b = b/b(1);

    # Function handle for r.
    def r(z):
        return mp.polyval(a[::-1], z) / mp.polyval(b[::-1], z)
         
    # Compute poles.
    poles_init = None
    for k in range(2):
        poles, err = mp.polyroots(b[::-1], maxsteps=5000, extraprec=100+10*k, error=True, roots_init=poles_init)
        print(f"{k=} {float(err)}, {float(max([mp.fabs(mp.polyval(b[::-1], z)) for z in poles]))}")
        if poles_init is not None:
            print([float(mp.fabs(p1 - p2)) for p1, p2 in zip(poles, poles_init)])
        poles_init = poles

    # Estimate residues.
    # t = 1e-7;  % Perturbation for residue estimate.
    # residues1 = t*(r(poles + t) - r(poles - t))/2

    # using c_n = p(z_n)/q'(z_n)
    bp = [bi * ni for bi, ni in zip(b[1:], n_range[1:])]
    def boqp(z):
        return mp.polyval(a[::-1], z) / mp.polyval(bp[::-1], z)
    residues = [boqp(pole) for pole in poles]

    r_inf = 0;
    if (m == n):
      r_inf = (-1)**m;

    return [r, a, b, poles, residues, r_inf]


if __name__ == "__main__":
    print(padephi_s(2, 2))
