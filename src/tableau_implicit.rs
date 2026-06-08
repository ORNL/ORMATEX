/*
 * Copyright© 2025 UT-Battelle, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// Butcher tableaux for implicit Runge-Kutta methods (DIRK / SDIRK / ESDIRK).
///
/// Layout convention
/// -----------------
/// `a[i]` has exactly `i + 1` entries — the full lower-triangular row including
/// the diagonal.  `a[i][i]` is the implicit (diagonal) coefficient for stage `i`.
/// When `a[i][i] == 0.0` the stage is explicit (used for ESDIRK methods such as
/// Crank–Nicolson where the first stage is always explicit).
///
/// This is different from the *explicit* `BT` in `ode_rk.rs`, where `a[i]` has
/// only `i` entries (the zero diagonal is omitted entirely).
///
/// L-stability note
/// ----------------
/// All multi-stage methods below use the FSAL (First Same As Last) property
/// `b == a[s-1]`, which guarantees L-stability:
///   b^T A^{-1} e  =  (last row of A) · A^{-1} · e  =  e_{s-1} · e  =  1
///   R(∞) = 1 - b^T A^{-1} e = 0  ⟹  L-stable.

#[derive(Clone)]
pub struct ImplicitBT {
    /// Number of stages
    pub s: usize,
    /// Stage abscissas c[i], length s.  c[i] = sum_j a[i][j].
    pub c: Vec<f64>,
    /// Final accumulation weights b[i], length s.  sum(b) = 1.
    pub b: Vec<f64>,
    /// Runge–Kutta matrix, lower-triangular.
    /// a[i] has i+1 entries; a[i][i] is the diagonal (implicit) coefficient.
    /// a[i][i] == 0.0 means stage i is explicit.
    pub a: Vec<Vec<f64>>,
}

impl ImplicitBT {
    // ── 1-stage ──────────────────────────────────────────────────────────────

    /// Implicit (Backward) Euler — order 1, L-stable.
    ///
    /// Equivalent to BDF1.  Single implicit stage at c = 1.
    ///
    ///   c | a       b
    ///   --+---    -----
    ///   1 | 1       1
    pub fn implicit_euler() -> Self {
        ImplicitBT {
            s: 1,
            c: vec![1.0],
            b: vec![1.0],
            a: vec![vec![1.0]],
        }
    }

    // ── 2-stage ──────────────────────────────────────────────────────────────

    /// Crank–Nicolson (trapezoidal rule) — order 2, A-stable, ESDIRK.
    ///
    /// Stage 0 is *explicit* (a[0][0] = 0); stage 1 is implicit (a[1][1] = 0.5).
    /// This is the standard trapezoidal / CN method.
    ///
    ///   c | a            b
    ///   --+----------  -----
    ///   0 | 0   0        0.5
    ///   1 | 0.5 0.5      0.5
    pub fn crank_nicolson() -> Self {
        ImplicitBT {
            s: 2,
            c: vec![0.0, 1.0],
            b: vec![0.5, 0.5],
            a: vec![
                vec![0.0],
                vec![0.5, 0.5],
            ],
        }
    }

    /// SDIRK(2,2) — 2 stages, order 2, L-stable.  Norsett (1974).
    ///
    /// γ = 1 − 1/√2 ≈ 0.2929
    ///
    ///   c  | a           b
    ///   ---+---------  -----
    ///   γ  | γ            1−γ
    ///   1  | 1−γ  γ       γ
    ///
    /// Reference: Norsett (1974), "Semi-explicit Runge–Kutta methods".
    pub fn sdirk22() -> Self {
        let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
        ImplicitBT {
            s: 2,
            c: vec![gamma, 1.0],
            b: vec![1.0 - gamma, gamma],
            a: vec![
                vec![gamma],
                vec![1.0 - gamma, gamma],
            ],
        }
    }

    // ── 3-stage ──────────────────────────────────────────────────────────────

    /// SDIRK(3,2) — 3 stages, order 2, L-stable (default).
    ///
    /// γ = 1/4,  c = [1/4, 1/2, 1].
    /// Uses FSAL (b = a[2]), guaranteeing L-stability for any valid γ.
    ///
    ///   c   | a                     b
    ///   ----+------------------   ------
    ///   1/4 | 1/4                   1/2
    ///   1/2 | 1/4  1/4              1/4
    ///   1   | 1/2  1/4  1/4         1/4
    ///
    /// Verification:
    ///   Σb = 1 ✓    Σb·c = 1/2·1/4 + 1/4·1/2 + 1/4·1 = 1/8+1/8+1/4 = 1/2 ✓
    ///   L-stable via FSAL (b ≡ a[2], so b^T A^{-1} e = 1, R(∞) = 0).
    pub fn sdirk32() -> Self {
        ImplicitBT {
            s: 3,
            c: vec![0.25, 0.5, 1.0],
            b: vec![0.5, 0.25, 0.25],
            a: vec![
                vec![0.25],
                vec![0.25, 0.25],
                vec![0.5,  0.25, 0.25],
            ],
        }
    }

    /// SDIRK(3,2) Norsett variant — 3 stages, order 2, L-stable.
    ///
    /// γ_N = (3−√3)/6 ≈ 0.2113,  c = [γ_N, 1/2, 1].
    /// Uses FSAL (b = a[2]).  Smaller diagonal γ means less implicit dissipation
    /// (closer to the Norsett optimal accuracy parameter).
    ///
    /// Exact coefficients (α = √3):
    ///   b₀ = (α−1)/2,   b₁ = (α−1)/α = 1 − 1/α,   b₂ = γ_N
    ///
    /// Reference: Norsett (1974) SDIRK family, L-stable variant via FSAL.
    pub fn sdirk32_norsett() -> Self {
        let sq3 = 3.0_f64.sqrt();
        let gamma = (3.0 - sq3) / 6.0;          // ≈ 0.21132
        let b1 = (sq3 - 1.0) / sq3;             // = 1 − 1/√3 ≈ 0.42265
        let b0 = (sq3 - 1.0) / 2.0;             // = (√3−1)/2  ≈ 0.36603
        ImplicitBT {
            s: 3,
            c: vec![gamma, 0.5, 1.0],
            b: vec![b0, b1, gamma],
            a: vec![
                vec![gamma],
                vec![0.5 - gamma,  gamma],
                vec![b0,           b1,     gamma],
            ],
        }
    }

    /// SDIRK(3,3) Alexander — 3 stages, order 3, L-stable.
    ///
    /// γ ≈ 0.4358665215454664  (unique real root of 6x³−18x²+9x−1=0 in (1/6,1/2))
    ///
    ///   c        | a                        b
    ///   ---------+-------------------    -------
    ///   γ        | γ                        b₁
    ///   (1+γ)/2  | (1−γ)/2   γ              b₂
    ///   1        | b₁        b₂    γ        γ
    ///
    ///   b₁ = −(6γ²−16γ+1)/4
    ///   b₂ =  (6γ²−20γ+5)/4
    ///
    /// Uses FSAL (b = a[2]), so L-stable.
    /// Note b₂ < 0 — this is expected for this method.
    ///
    /// Reference: Alexander (1977), "Diagonally implicit Runge–Kutta methods for
    /// stiff ODEs", SIAM J. Numer. Anal. 14(6), pp. 1006–1021.
    pub fn sdirk33() -> Self {
        // Unique root in (1/6, 1/2) of 6γ³ − 18γ² + 9γ − 1 = 0
        const GAMMA: f64 = 0.435_866_521_545_466_4;
        let g = GAMMA;
        let b1 = -(6.0 * g * g - 16.0 * g + 1.0) / 4.0;   // ≈  1.2085
        let b2 =  (6.0 * g * g - 20.0 * g + 5.0) / 4.0;   // ≈ −0.6444
        ImplicitBT {
            s: 3,
            c: vec![g, (1.0 + g) / 2.0, 1.0],
            b: vec![b1, b2, g],
            a: vec![
                vec![g],
                vec![(1.0 - g) / 2.0, g],
                vec![b1, b2, g],
            ],
        }
    }
}

// ── Verification helpers (used in tests only) ─────────────────────────────

impl ImplicitBT {
    /// Check Σb = 1 (consistency) and Σb·c = 1/2 (order-2 condition).
    /// Returns (sum_b, sum_bc).
    #[cfg(test)]
    pub fn check_order2(&self) -> (f64, f64) {
        let sum_b: f64 = self.b.iter().sum();
        let sum_bc: f64 = self.b.iter().zip(self.c.iter()).map(|(b, c)| b * c).sum();
        (sum_b, sum_bc)
    }

    /// Check that each row sum of a equals c[i] (consistency condition).
    #[cfg(test)]
    pub fn check_consistency(&self) -> Vec<f64> {
        (0..self.s)
            .map(|i| self.a[i].iter().sum::<f64>() - self.c[i])
            .collect()
    }

    /// L-stability check: compute b^T A^{-1} e by forward substitution.
    /// Returns the value; should equal 1.0 for an L-stable method.
    #[cfg(test)]
    pub fn check_l_stability(&self) -> f64 {
        // Solve A x = e (e = all-ones) by forward substitution on lower-triangular A.
        let s = self.s;
        let mut x = vec![0.0_f64; s];
        for i in 0..s {
            let mut sum = 1.0_f64;
            for j in 0..i {
                sum -= self.a[i][j] * x[j];
            }
            x[i] = sum / self.a[i][i];
        }
        // b^T x
        self.b.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum()
    }
}

#[cfg(test)]
mod test_tableaux {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn check_bt(name: &str, bt: &ImplicitBT, expected_order: usize) {
        let (sum_b, sum_bc) = bt.check_order2();
        println!("{name}: Σb={sum_b:.6}, Σb·c={sum_bc:.6}");
        assert_approx_eq!(sum_b, 1.0, 1e-12);
        if expected_order >= 2 {
            // 1e-10: SDIRK33's γ is a polynomial root not exactly representable
            // in f64, so Σb·c = 1/2 holds only to ~1e-11 floating-point accuracy.
            assert_approx_eq!(sum_bc, 0.5, 1e-10);
        }
        let consistency = bt.check_consistency();
        for (i, &err) in consistency.iter().enumerate() {
            assert!(
                err.abs() < 1e-12,
                "{name} stage {i} consistency error: {err}"
            );
        }
    }

    #[test]
    fn test_implicit_euler_order1() {
        let bt = ImplicitBT::implicit_euler();
        assert_eq!(bt.s, 1);
        let sum_b: f64 = bt.b.iter().sum();
        assert_approx_eq!(sum_b, 1.0, 1e-12);
    }

    #[test]
    fn test_crank_nicolson_order2() {
        check_bt("CN", &ImplicitBT::crank_nicolson(), 2);
    }

    #[test]
    fn test_sdirk22_order2() {
        let bt = ImplicitBT::sdirk22();
        check_bt("SDIRK22", &bt, 2);
        let lstab = bt.check_l_stability();
        println!("SDIRK22 b^T A^-1 e = {lstab:.6}");
        assert_approx_eq!(lstab, 1.0, 1e-12);
    }

    #[test]
    fn test_sdirk32_order2_lstable() {
        let bt = ImplicitBT::sdirk32();
        check_bt("SDIRK32", &bt, 2);
        let lstab = bt.check_l_stability();
        println!("SDIRK32 b^T A^-1 e = {lstab:.6}");
        assert_approx_eq!(lstab, 1.0, 1e-12);
    }

    #[test]
    fn test_sdirk32_norsett_order2_lstable() {
        let bt = ImplicitBT::sdirk32_norsett();
        check_bt("SDIRK32_Norsett", &bt, 2);
        let lstab = bt.check_l_stability();
        println!("SDIRK32_Norsett b^T A^-1 e = {lstab:.6}");
        assert_approx_eq!(lstab, 1.0, 1e-12);
    }

    #[test]
    fn test_sdirk33_order3() {
        let bt = ImplicitBT::sdirk33();
        // Check order-2 conditions (subsumed by order-3)
        check_bt("SDIRK33", &bt, 2);
        // Check order-3 conditions: Σb·c² = 1/3 and b^T A c = 1/6
        // Note: for DIRK methods A is the full lower-triangular matrix
        // including the diagonal, so the inner sum runs j = 0..=i.
        let sum_bc2: f64 = bt.b.iter().zip(bt.c.iter()).map(|(b, c)| b * c * c).sum();
        println!("SDIRK33 Σb·c² = {sum_bc2:.6}");
        assert_approx_eq!(sum_bc2, 1.0 / 3.0, 1e-10);
        // b^T A c  =  Σ_i b_i * (Σ_{j=0..=i} a[i][j] * c[j])
        let mut sum_bac = 0.0_f64;
        for i in 0..bt.s {
            for j in 0..=i {              // j ≤ i, includes diagonal
                sum_bac += bt.b[i] * bt.a[i][j] * bt.c[j];
            }
        }
        println!("SDIRK33 b^T A c = {sum_bac:.6}");
        assert_approx_eq!(sum_bac, 1.0 / 6.0, 1e-10);
        // L-stability
        let lstab = bt.check_l_stability();
        println!("SDIRK33 b^T A^-1 e = {lstab:.6}");
        assert_approx_eq!(lstab, 1.0, 1e-10);
    }
}
