use ormatex::mat_utils::mat_mat_approx_eq;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_leja::*;
use ormatex::ode_sys::DynRefExtendedLinOp;
use ormatex::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use ormatex::matexp_pade::{matexp, PadeExpm};
use faer::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Parse a Matrix Market (.mtx) file and return (rows, cols, triplets as (i,j,val))
fn read_mtx_file(path: &str) -> (usize, usize, Vec<(usize, usize, f64)>) {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header line (%%MatrixMarket matrix coordinate real general)
    let _header = lines.next();

    // Skip comment lines (start with %)
    loop {
        let line = lines.next().expect("No dimensions line found").expect("Failed to read line");
        if !line.trim().starts_with("%") {
            // Parse dimensions line: rows cols nnz
            let dims: Vec<usize> = line
                .split_whitespace()
                .map(|s| s.parse::<usize>().expect("Failed to parse dimension"))
                .collect();
            let rows = dims[0];
            let cols = dims[1];

            // Parse triplets
            let mut triplets = Vec::new();
            for line in lines {
                let line = line.expect("Failed to read line");
                if line.trim().is_empty() {
                    continue;
                }
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    // 1-indexed to 0-indexed
                    let i = parts[0].parse::<usize>().expect("Failed to parse row") - 1;
                    let j = parts[1].parse::<usize>().expect("Failed to parse col") - 1;
                    let val = parts[2].parse::<f64>().expect("Failed to parse value");
                    triplets.push((i, j, val));
                }
            }

            return (rows, cols, triplets);
        }
    }
}

/// Regression test for Leja phi_k vector evaluator
///
/// This test verifies that the Leja polynomial method for computing
/// phi_k(A)*v products works correctly by comparing against the Krylov and Pade methods.
///
/// Args
/// * `krylov_reuse` - reuse the ritz values and hessenberg from arnoldi
///                    in the leja polynomial approximation
/// * `k` - phi function order
///
fn _case_s3_phikv(krylov_reuse: bool, k: usize)
{
    // Load Jacobian matrix from test_data/s3_jacobian.mtx
    let (m_rows, m_cols, jac_triplets_raw) = read_mtx_file("tests/test_data/s3_jacobian.mtx");
    // Convert to faer triplets
    let jac_triplets: Vec<_> = jac_triplets_raw
        .iter()
        .map(|(i, j, val)| faer::sparse::Triplet::new(*i, *j, *val))
        .collect();
    let jac_sparse = SparseColMat::<usize, f64>::try_new_from_triplets(m_rows, m_cols, &jac_triplets)
        .expect("Failed to create sparse matrix");

    // Load y vector from test_data/s3_y.mtx
    let (y_rows, _y_cols, y_triplets_raw) = read_mtx_file("tests/test_data/s3_y.mtx");
    let mut y_vec = Mat::<f64>::zeros(y_rows, 1);
    for (i, _j, val) in y_triplets_raw {
        y_vec[(i, 0)] = val;
    }

    let dt = 0.5;

    // Convert sparse matrix to dense for comparison
    let jac_dense = jac_sparse.as_ref().to_dense();

    // Compute reference solution using Pade method
    let expmv = Box::new(PadeExpm::new(12));
    let pade_phikv = expmv.phik_apply(jac_dense.as_ref(), dt, y_vec.as_ref(), k);

    // Compute using Leja evaluator
    let lp = LejaPoints::new_from_fn("leja_circle").slice(0, 400);

    // Setup Arnoldi-based spectrum estimation
    let leja_ellipse_adapter = LejaEllipseAdapterArnoldiIOM::new(
        -1.0, 0.0, 1.0, 1e-8, 24, 2, 1.05);

    let mut leja_eval = LejaPhiEval::new(
        lp, 400, 1e-16, "clapm", "dd_taylor", krylov_reuse,
        Box::new(leja_ellipse_adapter));

    // Build vb_vec and ext_jac_lo before apply_prepare so the correct BAMPHI
    // Arnoldi starting vector (upper_block(ext^p * tilde_v)) can be computed.
    let mut zero_vec = y_vec.clone();
    zero_vec.fill(0.0);
    let vb_vec = if k == 1 {
            vec![zero_vec.as_ref(), y_vec.as_ref()]
        } else if k == 2 {
            vec![zero_vec.as_ref(), zero_vec.as_ref(), y_vec.as_ref()]
        } else {
            vec![y_vec.as_ref()]
        };
    let ext_jac_lo = DynRefExtendedLinOp::new(dt, &jac_sparse, &vb_vec);

    // Prepare the evaluator with the correct extended operator
    leja_eval.apply_prepare(&jac_sparse, dt, y_vec.as_ref(), k, Some((&ext_jac_lo, &vb_vec)));

    println!("n_ritz: {}", leja_eval.leja_ellipse_adapter.n_ritz());
    println!("leja ritz_re: {:?}", leja_eval.leja_ellipse_adapter.get_ritz().0.unwrap());
    println!("leja ritz_im: {:?}", leja_eval.leja_ellipse_adapter.get_ritz().1.unwrap());

    // Apply phi_0 using Leja polynomial method
    let leja_phikv = leja_eval.apply_phi_k_v(&ext_jac_lo, 1.0, &vb_vec);

    // Verify result is finite and reasonable
    assert_eq!(leja_phikv.nrows(), y_rows, "Result should have same number of rows as y");
    assert_eq!(leja_phikv.ncols(), 1, "Result should be a column vector");
    let result_norm = leja_phikv.norm_l2();
    assert!(result_norm.is_finite(), "Result norm should be finite");
    assert!(result_norm > 0.0, "Result should be non-zero");

    // setup krylov phi evaluator
    let iom = 2;
    let tol = 1e-12;
    let m = 80;
    let krylov_dim_max = 400;
    let mut krylov_phikv_eval = KrylovExpm::new(expmv, m, krylov_dim_max, tol, Some(iom));

    // Apply phi_0 using Krylov method
    let krylov_phikv: Mat<f64> = krylov_phikv_eval.apply_phi_k_v(&ext_jac_lo, 1.0, &vb_vec);

    // Verify numerical consistency with krylov
    // Compute relative error
    let diff_norm = (leja_phikv.as_ref() - krylov_phikv.as_ref()).norm_l2();
    let krylov_norm = krylov_phikv.norm_l2();
    let relative_error = diff_norm / krylov_norm;
    println!("Leja result norm: {:0.6e}", leja_phikv.norm_l2());
    println!("Krylov result norm: {:0.6e}", krylov_norm);
    println!("Relative error: {:0.6e}", relative_error);
    assert!(relative_error < 1e-8);
    mat_mat_approx_eq(leja_phikv.as_ref(), krylov_phikv.as_ref(), 1e-6);

    // Verify numerical consistency with pade
    // Compute relative error
    let diff_norm = (leja_phikv.as_ref() - pade_phikv.as_ref()).norm_l2();
    let pade_norm = pade_phikv.norm_l2();
    let relative_error = diff_norm / pade_norm;
    println!("Pade result norm: {:0.6e}", pade_norm);
    println!("Relative error: {:0.6e}", relative_error);
    assert!(relative_error < 1e-8);
    mat_mat_approx_eq(leja_phikv.as_ref(), pade_phikv.as_ref(), 1e-6);
}

/// Tests phi_0(A)*v products
#[test]
fn test_case_s3_phi0v() {
    _case_s3_phikv(false, 0);
}
#[test]
fn test_case_s3_phi0v_krylov_reuse() {
    // enable krylov reuse
    _case_s3_phikv(true, 0);
}

/// Tests phi_1(A)*v products
#[test]
fn test_case_s3_phi1v() {
    _case_s3_phikv(false, 1);
}
#[test]
fn test_case_s3_phi1v_krylov_reuse() {
    // enable krylov reuse
    _case_s3_phikv(true, 1);
}

/// Tests phi_2(A)*v products
#[test]
fn test_case_s3_phi2v() {
    _case_s3_phikv(false, 2);
}
#[test]
fn test_case_s3_phi2v_krylov_reuse() {
    // enable krylov reuse
    _case_s3_phikv(true, 2);
}
