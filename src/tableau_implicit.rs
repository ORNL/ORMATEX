/*
 * Copyright(c) 2025 UT-Battelle, LLC
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
//! Butcher tableaux for diagonally implicit Runge-Kutta methods.
//!
//! Provides [`ImplicitBT`], the Butcher tableau of a DIRK / SDIRK / ESDIRK
//! method, together with constructors for the methods used by the implicit
//! integrators in [`crate::ode_implicit`]: implicit Euler, Crank-Nicolson,
//! and several L-stable SDIRK methods. An $s$-stage method advances
//! $y^\prime  = f(t, y)$ by
//!
//! $$ Y_i = y_n + \Delta t \sum_{j=1}^{i} a_{ij} f(t_n + c_j \Delta t, Y_j),
//! \qquad y_{n+1} = y_n + \Delta t \sum_{i=1}^{s} b_i f(t_n + c_i \Delta t, Y_i) $$
//!
//! The method is defined by its Butcher tableau, with the stage times $c$, the
//! lower-triangular matrix $A = (a_{ij})$ and the weights $b$:
//!
//! $$ \begin{array}{c|c} c & A \cr \hline & b^T \end{array} $$
//!
//! # FSAL and stiff accuracy
//!
//! A method has the FSAL (First Same As Last) property when the last row of $A$
//! equals the weight vector and the last stage time is one:
//!
//! $$ a_{sj} = b_j \quad (j = 1, \ldots, s), \qquad c_s = 1 . $$
//!
//! Then the last stage value is the new solution, $Y_s = y_{n+1}$, and the last
//! stage derivative $f(t_{n+1}, Y_s)$ is the derivative at the start of the next
//! step. For explicit and ESDIRK methods that derivative can be reused as the
//! first stage of the next step, saving one evaluation of $f$. For diagonally
//! implicit methods the same condition is called stiff accuracy. If in addition
//! $A$ is invertible, the stability function satisfies
//!
//! $$ R(\infty) = 1 - b^T A^{-1} e = 0, \qquad e = (1, \ldots, 1)^T, $$
//!
//! so an A-stable method is L-stable, which damps stiff components completely.
//! It also means the method is well suited to differential-algebraic systems,
//! as the algebraic constraints hold exactly at the end of the step. Note that
//! the stage implementation in this crate does not currently reuse the last
//! stage derivative in the next step.
//!
//! # References
//!
//! * Norsett, S. P., "Semi-explicit Runge-Kutta methods", Report Mathematics and
//!   Computation No. 6/74, Department of Mathematics, University of Trondheim, 1974
//! * Alexander, R., "Diagonally implicit Runge-Kutta methods for stiff O.D.E.'s",
//!   SIAM J. Numer. Anal. 14(6) (1977) 1006-1021

/// Butcher tableau for implicit Runge-Kutta methods (DIRK / SDIRK / ESDIRK).
///
/// # Layout convention
///
/// `a[i]` has exactly `i + 1` entries - the full lower-triangular row including
/// the diagonal.  `a[i][i]` is the implicit (diagonal) coefficient for stage `i`.
/// When `a[i][i] == 0.0` the stage is explicit (used for ESDIRK methods such as
/// Crank-Nicolson where the first stage is always explicit).
///
/// This is different from the explicit `BT` in `ode_rk.rs`, where `a[i]` has
/// only `i` entries (the zero diagonal is omitted entirely).
///
#[derive(Clone)]
pub struct ImplicitBT {
    /// Number of stages
    pub s: usize,
    /// Stage abscissas `c[i]`, length `s`, with $c_i = \sum_j a_{ij}$.
    pub c: Vec<f64>,
    /// Final accumulation weights `b[i]`, length `s`, with $\sum_i b_i = 1$.
    pub b: Vec<f64>,
    /// Runge-Kutta matrix, lower-triangular.
    /// `a[i]` has i+1 entries; `a[i][i]` is the diagonal (implicit) coefficient.
    /// `a[i][i]` == 0.0 means stage i is explicit.
    pub a: Vec<Vec<f64>>,
}

impl ImplicitBT {
    /// Implicit (Backward) Euler, order 1, L-stable.
    ///
    /// Equivalent to BDF1. It has a single implicit stage at $c_1 = 1$:
    ///
    /// $$ \begin{array}{c|c} 1 & 1 \cr \hline & 1 \end{array} $$
    pub fn implicit_euler() -> Self {
        ImplicitBT {
            s: 1,
            c: vec![1.0],
            b: vec![1.0],
            a: vec![vec![1.0]],
        }
    }

    /// Crank-Nicolson (trapezoidal rule), order 2, A-stable, ESDIRK.
    ///
    /// The first stage is explicit ($a_{11} = 0$) and the second stage is
    /// implicit ($a_{22} = 1/2$). This is the standard trapezoidal method:
    ///
    /// $$ \begin{array}{c|cc}
    /// 0 & 0 & 0 \cr
    /// 1 & 1/2 & 1/2 \cr \hline
    ///   & 1/2 & 1/2
    /// \end{array} $$
    ///
    /// Because the first stage is explicit, $A$ is singular and the method is not
    /// L-stable. Note that `a[0]` stores only the single entry `0.0` (the diagonal).
    pub fn crank_nicolson() -> Self {
        ImplicitBT {
            s: 2,
            c: vec![0.0, 1.0],
            b: vec![0.5, 0.5],
            a: vec![vec![0.0], vec![0.5, 0.5]],
        }
    }

    /// SDIRK(2,2), 2 stages, order 2, L-stable. Norsett (1974).
    ///
    /// The diagonal coefficient is $\gamma = 1 - 1/\sqrt{2} \approx 0.2929$:
    ///
    /// $$ \begin{array}{c|cc}
    /// \gamma & \gamma & 0 \cr
    /// 1 & 1 - \gamma & \gamma \cr \hline
    ///   & 1 - \gamma & \gamma
    /// \end{array} $$
    ///
    /// The method is FSAL (see the module documentation).
    ///
    /// # References
    ///
    /// * Norsett, S. P., "Semi-explicit Runge-Kutta methods", Report Mathematics
    ///   and Computation No. 6/74, University of Trondheim, 1974
    pub fn sdirk22() -> Self {
        let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
        ImplicitBT {
            s: 2,
            c: vec![gamma, 1.0],
            b: vec![1.0 - gamma, gamma],
            a: vec![vec![gamma], vec![1.0 - gamma, gamma]],
        }
    }

    /// SDIRK(3,2), 3 stages, order 2, L-stable (default).
    ///
    /// The diagonal coefficient is $\gamma = 1/4$ and $c = (1/4, 1/2, 1)$.
    /// The method is FSAL, $b = $ `a[2]` (see the module documentation), which
    /// gives $R(\infty) = 0$ and hence L-stability.
    ///
    /// $$ \begin{array}{c|ccc}
    /// 1/4 & 1/4 & 0 & 0 \cr
    /// 1/2 & 1/4 & 1/4 & 0 \cr
    /// 1 & 1/2 & 1/4 & 1/4 \cr \hline
    ///   & 1/2 & 1/4 & 1/4
    /// \end{array} $$
    pub fn sdirk32() -> Self {
        ImplicitBT {
            s: 3,
            c: vec![0.25, 0.5, 1.0],
            b: vec![0.5, 0.25, 0.25],
            a: vec![vec![0.25], vec![0.25, 0.25], vec![0.5, 0.25, 0.25]],
        }
    }

    /// SDIRK(3,2) Norsett variant, 3 stages, order 2, L-stable.
    ///
    /// The diagonal coefficient is $\gamma_N = (3 - \sqrt{3})/6$ and
    /// $c = (\gamma_N, 1/2, 1)$. The method is FSAL, $b = $ `a[2]` (see the
    /// module documentation). The smaller diagonal $\gamma_N$ means less
    /// implicit dissipation (closer to the Norsett optimal accuracy parameter).
    ///
    /// With $\alpha = \sqrt{3}$ the weights are
    ///
    /// $$ b_1 = \frac{\alpha - 1}{2}, \qquad
    /// b_2 = \frac{\alpha - 1}{\alpha} = 1 - \frac{1}{\alpha}, \qquad
    /// b_3 = \gamma_N , $$
    ///
    /// and the tableau is
    ///
    /// $$ \begin{array}{c|ccc}
    /// \gamma_N & \gamma_N & 0 & 0 \cr
    /// 1/2 & 1/2 - \gamma_N & \gamma_N & 0 \cr
    /// 1 & b_1 & b_2 & \gamma_N \cr \hline
    ///   & b_1 & b_2 & \gamma_N
    /// \end{array} $$
    ///
    /// # References
    ///
    /// * Norsett, S. P., "Semi-explicit Runge-Kutta methods", Report Mathematics
    ///   and Computation No. 6/74, University of Trondheim, 1974 (SDIRK family,
    ///   L-stable variant via FSAL)
    pub fn sdirk32_norsett() -> Self {
        let sq3 = 3.0_f64.sqrt();
        let gamma = (3.0 - sq3) / 6.0;
        let b1 = (sq3 - 1.0) / sq3;
        let b0 = (sq3 - 1.0) / 2.0;
        ImplicitBT {
            s: 3,
            c: vec![gamma, 0.5, 1.0],
            b: vec![b0, b1, gamma],
            a: vec![vec![gamma], vec![0.5 - gamma, gamma], vec![b0, b1, gamma]],
        }
    }

    /// SDIRK(3,3) Alexander, 3 stages, order 3, L-stable.
    ///
    /// The diagonal coefficient $\gamma \approx 0.4358665215454664$ is a root of
    /// $\gamma^3 - 3 \gamma^2 + \frac{3}{2} \gamma - \frac{1}{6} = 0$. The weights are
    ///
    /// $$ b_1 = -\frac{6 \gamma^2 - 16 \gamma + 1}{4}, \qquad
    /// b_2 = \frac{6 \gamma^2 - 20 \gamma + 5}{4}, $$
    ///
    /// and the tableau is
    ///
    /// $$ \begin{array}{c|ccc}
    /// \gamma & \gamma & 0 & 0 \cr
    /// (1 + \gamma)/2 & (1 - \gamma)/2 & \gamma & 0 \cr
    /// 1 & b_1 & b_2 & \gamma \cr \hline
    ///   & b_1 & b_2 & \gamma
    /// \end{array} $$
    ///
    /// The method is FSAL (see the module documentation).
    ///
    /// # References
    ///
    /// * Alexander, R., "Diagonally implicit Runge-Kutta methods for stiff O.D.E.'s",
    ///   SIAM J. Numer. Anal. 14(6) (1977) 1006-1021
    pub fn sdirk33() -> Self {
        const GAMMA: f64 = 0.435_866_521_545_466_4;
        let g = GAMMA;
        let b1 = -(6.0 * g * g - 16.0 * g + 1.0) / 4.0;
        let b2 = (6.0 * g * g - 20.0 * g + 5.0) / 4.0;
        ImplicitBT {
            s: 3,
            c: vec![g, (1.0 + g) / 2.0, 1.0],
            b: vec![b1, b2, g],
            a: vec![vec![g], vec![(1.0 - g) / 2.0, g], vec![b1, b2, g]],
        }
    }
}

impl ImplicitBT {
    /// Check $\sum_i b_i = 1$ (consistency) and $\sum_i b_i c_i = 1/2$ (order-2 condition).
    /// Returns `(sum_b, sum_bc)`.
    #[cfg(test)]
    pub fn check_order2(&self) -> (f64, f64) {
        let sum_b: f64 = self.b.iter().sum();
        let sum_bc: f64 = self.b.iter().zip(self.c.iter()).map(|(b, c)| b * c).sum();
        (sum_b, sum_bc)
    }

    /// Check that each row sum of $A$ equals `c[i]` (consistency condition).
    #[cfg(test)]
    pub fn check_consistency(&self) -> Vec<f64> {
        (0..self.s)
            .map(|i| self.a[i].iter().sum::<f64>() - self.c[i])
            .collect()
    }

    /// L-stability check: compute $b^T A^{-1} e$ by forward substitution.
    /// Returns the value, which should equal 1 for an L-stable method.
    ///
    /// # L-stability
    ///
    /// The stability function at infinity is $R(\infty) = 1 - b^T A^{-1} e$.
    /// The multi-stage methods above are FSAL (see the module documentation), so
    /// $b^T$ is the last row of $A$ and
    ///
    /// $$ b^T A^{-1} e = e_s^T A A^{-1} e = 1 , $$
    ///
    /// hence $R(\infty) = 0$.
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
mod test_tableau {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn check_bt(name: &str, bt: &ImplicitBT, expected_order: usize) {
        let (sum_b, sum_bc) = bt.check_order2();
        println!("{name}: Σb={sum_b:.6}, Σb·c={sum_bc:.6}");
        assert_approx_eq!(sum_b, 1.0, 1e-12);
        if expected_order >= 2 {
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
        // Check order-3 conditions: sum(b*c^2) = 1/3 and b^T A c = 1/6
        // Note: for DIRK methods A is the full lower-triangular matrix
        // including the diagonal, so the inner sum runs j = 0..=i.
        let sum_bc2: f64 = bt.b.iter().zip(bt.c.iter()).map(|(b, c)| b * c * c).sum();
        println!("SDIRK33 Σb·c² = {sum_bc2:.6}");
        assert_approx_eq!(sum_bc2, 1.0 / 3.0, 1e-10);
        // b^T A c  =  sum_i b_i * (sum_{j=0..=i} a[i][j] * c[j])
        let mut sum_bac = 0.0_f64;
        for i in 0..bt.s {
            for j in 0..=i {
                // j <= i, includes diagonal
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
