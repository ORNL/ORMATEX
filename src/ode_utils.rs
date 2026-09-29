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
//! Test and example helpers: right hand sides of small ODE systems (Lotka-Volterra, Bateman, Robertson).
use faer::prelude::*;

/// Lotka-Volterra rhs (test use only) with all parameters 1: $u^\prime  = u - u v$, $v^\prime  = u v - v$ for $x = (u, v)$.
pub fn lv_sys_rhs(_t: f64, x: MatRef<f64>) -> Mat<f64> {
    let alpha = 1.0;
    let beta = 1.0;
    let delta = 1.0;
    let gamma = 1.0;

    faer::mat![
        [alpha * x[(0, 0)] - beta * x[(0, 0)] * x[(1, 0)]],
        [delta * x[(0, 0)] * x[(1, 0)] - gamma * x[(1, 0)]],
    ]
}

/// Sparse $2 \times 2$ Jacobian of [`lv_sys_rhs`] (test use only): $J = \begin{pmatrix} 1 - v & -u \cr v & u - 1 \end{pmatrix}$.
pub fn lv_sys_jac(_t: f64, x: MatRef<f64>) -> SparseColMat<usize, f64> {
    let alpha = 1.0;
    let beta = 1.0;
    let delta = 1.0;
    let gamma = 1.0;

    let jac = faer::mat![
        [alpha - beta * x[(1, 0)], -beta * x[(0, 0)]],
        [delta * x[(1, 0)], delta * x[(0, 0)] - gamma],
    ];
    // convert to sparse
    let mut jac_triplets = Vec::new();
    for i in 0..jac.nrows() {
        for j in 0..jac.ncols() {
            jac_triplets.push(faer::sparse::Triplet::new(i, j, jac[(i, j)]));
        }
    }
    let jac_sprs =
        SparseColMat::<usize, f64>::try_new_from_triplets(jac.nrows(), jac.ncols(), &jac_triplets)
            .unwrap();
    jac_sprs
}

// Bateman
/// Linear stiff Bateman-type rhs $x^\prime  = M x$ (test use only), best case for exponential integrators.
///
/// $M$ is upper bidiagonal with diagonal $(-10^{-3}, -10, -10^{-1})$ and superdiagonal $(10, 10^{-1})$.
pub fn bateman_sys_rhs(_t: f64, x: MatRef<f64>) -> Mat<f64> {
    // slow decay
    let lambda_0 = 1.0e-3;
    // fast decay
    let lambda_1 = 1.0e1;
    // med decay
    let lambda_2 = 1.0e-1;
    // near stable
    // let lambda_3 = 1.0e-16;

    let bat_mat = faer::mat![
        [-lambda_0, lambda_1, 0.],
        [0., -lambda_1, lambda_2],
        [0., 0., -lambda_2],
    ];

    let xdot = bat_mat * x.as_ref();
    xdot
}

/// Robertson stiff kinetics rhs (test use only): $x^\prime  = -0.04 x + 10^4 y z$, $y^\prime  = 0.04 x - 10^4 y z - 3 \cdot 10^7 y^2$, $z^\prime  = 3 \cdot 10^7 y^2$.
pub fn rob_sys_rhs(_t: f64, x_in: MatRef<f64>) -> Mat<f64> {
    let x = x_in[(0, 0)];
    let y = x_in[(1, 0)];
    let z = x_in[(2, 0)];

    let xdot = -0.04 * x + 1.0e4 * y * z;
    let ydot = 0.04 * x - 1.0e4 * y * z - 3.0e7 * y.powf(2.0);
    let zdot = 3.0e7 * y.powf(2.0);

    faer::mat![[xdot], [ydot], [zdot],]
}
