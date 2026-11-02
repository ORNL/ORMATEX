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
import numpy as np
import scipy as sp
from scipy.linalg import hankel, svd


def phiS(z, s):
    """Compute phi_s(Z), reliably up to S=10 or so."""
    if s == 0:
        return np.exp(z)

    z = np.atleast_1d(z)
    shape = z.shape
    z = z.reshape(-1)
    phi_s_z = np.zeros_like(z, dtype=complex if np.iscomplexobj(z) else float)

    def tayl(z_sub, N, S):
        # Broadcast elements over powers 0 to N
        powers = np.arange(N + 1)
        z_mesh, p_mesh = np.meshgrid(z_sub, powers, indexing='ij')
        terms = (z_mesh ** p_mesh) / sp.special.factorial(powers + S)
        return np.sum(terms, axis=1)

    Is = np.abs(z) < 1
    if np.any(Is):
        # for |z| < 1, a 20+ degree Taylor approximation works well.
        phi_s_z[Is] = tayl(z[Is], 20 + s, s)

    not_Is = ~Is
    if np.any(not_Is):
        # for |z| > 1, there is no meaningful cancellation in exp(z) - T_{s-1}(z).
        phi_s_z[not_Is] = (np.exp(z[not_Is]) - tayl(z[not_Is], s - 1, 0)) / (z[not_Is] ** s)

    ret = phi_s_z.item() if shape == (1,) else phi_s_z.reshape(shape)
    return ret


def cf_rat_phi(n, s):
    r"""Python implementation of the method from:
    Schmelzer, Thomas & Trefethen, Lloyd. (2007).
    Evaluating matrix functions for exponential integrators
    via Carathéodory–Fejér approximation and contour integrals. ETNA 29.
    """
    K = 75  # no of Cheb coeffs
    nf = 1024  # no of pts for FFT

    w = np.exp(2j * np.pi * np.arange(nf) / nf)  # roots of unity
    t = np.real(w)  # Cheb pts (twice over)
    scl = 9  # scale factor for stability

    F = np.zeros(t.shape, dtype=complex)
    g = (t != -1)

    # Evaluate Phi_S on neg. axis
    F[g] = phiS(scl * (t[g] - 1) / (t[g] + 1), s)

    c = np.real(np.fft.fft(np.real(F))) / nf  # Cheb coeffs of F

    # polyval reads coefficients highest-degree-first in numpy
    # c[K::-1] grabs c[K], c[K-1]... down to c[0]
    f = np.polyval(c[K::-1], w)  # analytic part f of F

    # SVD of Hankel matrix (MATLAB 1:K -> Python 0:K-1, indices 1:K+1 -> 1:K+1)
    H = hankel(c[1:K+1])
    U, S_vals, Vh = svd(H)
    V = Vh.conj().T  # scipy returns V^H, we need V

    sn = S_vals[n]  # singular value (0-based)

    # Extract singular vectors and reverse U to match MATLAB's K:-1:1
    u = U[K-1::-1, n]
    v = V[:, n]

    zz = np.zeros(nf - K)  # zeros for padding

    # Finite Blaschke product
    b = np.fft.fft(np.concatenate([u, zz])) / np.fft.fft(np.concatenate([v, zz]))

    rt = f - sn * (w ** K) * b  # extended function r-tilde
    # rtc = np.real(np.fft.fft(rt)) / nf  # Laurent coeffs (unused in original script)

    zr = np.roots(v)
    qj = zr[np.abs(zr) > 1]  # poles
    qc = np.poly(qj)  # coeffs of denominator

    pt = rt * np.polyval(qc, w)  # numerator
    ptc = np.real(np.fft.fft(pt) / nf)  # coeffs of numerator
    ptc = ptc[n::-1]  # Slice n down to 0

    ci = np.zeros(qj.shape, dtype=complex)
    for k in range(n):  # calculate residues
        q = qj[k]
        q2 = np.poly(qj[qj != q])
        ci[k] = np.polyval(ptc, q) / np.polyval(q2, q)

    zi = scl * (qj - 1) ** 2 / (qj + 1) ** 2  # poles in z-plane
    ci = 4 * ci * zi / (qj ** 2 - 1)  # residues in z-plane

    r_inf = 0.5 * (phiS(0, s) + np.sum(ci / zi))  # r at infinity

    # Sort poles by imaginary parts
    order = np.argsort(np.imag(zi))
    zi = zi[order]
    ci = ci[order]

    return zi, ci, r_inf.real
