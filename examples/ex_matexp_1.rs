/// Demo showing the evaluation the matrix exponential using pade approx
/// and partial fraction decomposition based methods.
use ormatex::matexp_cauchy;
use ormatex::matexp_pade;


pub fn main() {
    // example matrix
    let lmat = faer::mat![
        [-1.0e-3, 1.0e1, 0.],
        [0., -1.0e1, 1.0e-1],
        [0., 0., -1.0e-1],
    ];

    // expm(dt*L) with pade approx
    let dt = 1.0;
    let exp_lmat_pade = matexp_pade::matexp(lmat.as_ref(), dt);

    // expm(dt*L) with partial fraction decomposition method
    let order = 24;
    let matexp_eval = matexp_cauchy::gen_parabolic_expm(order);
    let exp_lmat_pdf = matexp_eval.matexp_dense_cauchy(lmat.as_ref(), dt);

    // show the results are consistent
    println!("Pade expm: {:?}", exp_lmat_pade.as_ref());
    println!("PFD expm: {:?}", exp_lmat_pdf.as_ref());
    println!(
        "norm(diff): {:?}",
        (exp_lmat_pdf.as_ref() - exp_lmat_pade.as_ref()).norm_l2()
    );

    // output plot storage
    let mut t_points: Vec<f64> = Vec::new();
    let mut c0: Vec<f64> = Vec::new();
    let mut c1: Vec<f64> = Vec::new();
    let mut c2: Vec<f64> = Vec::new();

    // A simple integration procedure for pure-linear systems
    // Step system forward in time with u_t+1 = expm(dt*lmat)*u_t
    let mut y = faer::mat![[0.001], [0.1], [1.0]];
    let mut t = 0.0;
    for _i in 0..10000 {
        y = matexp_eval.matexp_dense_cauchy(lmat.as_ref(), dt) * y.as_ref();
        t += dt;
        t_points.push(t);
        c0.push(y[(0, 0)]);
        c1.push(y[(1, 0)]);
        c2.push(y[(2, 0)]);
    }
}
