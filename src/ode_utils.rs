/// Useful tools for testing ODE integration methods
use faer::prelude::*;
use faer::sparse::*;


/// define Lotka-Volterra system for testing ONLY
pub fn lv_sys_rhs(t: f64, x: MatRef<f64>) -> Mat<f64> {
    let alpha = 1.0;
    let beta = 1.0;
    let delta = 1.0;
    let gamma = 1.0;

    faer::mat![
        [alpha * x[(0, 0)] - beta * x[(0, 0)]*x[(1, 0)] ],
        [delta * x[(0, 0)]*x[(1, 0)] - gamma * x[(1, 0)] ],
    ]
}

/// define Lotka-Volterra jacobian for testing ONLY
pub fn lv_sys_jac(t: f64, x: MatRef<f64>) -> SparseColMat<usize, f64> {
    let alpha = 1.0;
    let beta = 1.0;
    let delta = 1.0;
    let gamma = 1.0;

    let jac = faer::mat![
        [alpha - beta*x[(1, 0)], -beta*x[(0, 0)] ],
        [delta*x[(1, 0)], delta*x[(0, 0)] - gamma ],
    ];
    // convert to sparse
    let mut jac_triplets = Vec::new();
    for i in 0..jac.nrows() {
        for j in 0..jac.ncols() {
            jac_triplets.push(faer::sparse::Triplet::new(i, j, jac[(i, j)]));
        }
    }
    let jac_sprs = SparseColMat::<usize, f64>::try_new_from_triplets(
        jac.nrows(), jac.ncols(), &jac_triplets).unwrap();
    jac_sprs
}

// Bateman
/// Linear stiff.  Best-case scinario for ETD methods
pub fn bateman_sys_rhs(t: f64, x: MatRef<f64>) -> Mat<f64> {
    // slow decay
    let lambda_0 = 1.0e-3;
    // fast decay
    let lambda_1 = 1.0e1;
    // med decay
    let lambda_2 = 1.0e-1;
    // near stable
    // let lambda_3 = 1.0e-16;

    let n0 = x[(0, 0)];
    let n1 = x[(1, 0)];
    let n2 = x[(2, 0)];

    let bat_mat = faer::mat![
        [-lambda_0,  lambda_1,        0.],
        [       0., -lambda_1,  lambda_2],
        [       0.,        0., -lambda_2],
    ];

    let xdot = bat_mat * x.as_ref();
    xdot
}

/// Robertson
/// Nonlinear stiff example system.
pub fn rob_sys_rhs(t: f64, x_in: MatRef<f64>) -> Mat<f64> {
    let x = x_in[(0, 0)];
    let y = x_in[(1, 0)];
    let z = x_in[(2, 0)];

    let xdot = -0.04 * x + 1.0e4 * y * z;
    let ydot = 0.04 * x - 1.0e4 * y * z - 3.0e7 * y.powf(2.0);
    let zdot = 3.0e7 * y.powf(2.0);

    faer::mat![
        [xdot],
        [ydot],
        [zdot],
    ]
}
