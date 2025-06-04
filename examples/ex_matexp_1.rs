/// Demo showing the evaluation the matrix exponential using pade approx
/// and partial fraction decomposition based methods.
use faer::prelude::*;
use ormatex::matexp_pade;
use ormatex::matexp_cauchy;
#[cfg(feature = "plotters")]
use plotters::prelude::*;


pub fn main() {
    // example matrix
    let lmat = faer::mat![
        [-1.0e-3,    1.0e1,      0.],
        [       0., -1.0e1,  1.0e-1],
        [       0.,     0., -1.0e-1],
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
    println!("norm(diff): {:?}", (exp_lmat_pdf.as_ref() - exp_lmat_pade.as_ref()).norm_l2());

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
        y = matexp_eval.matexp_dense_cauchy(lmat.as_ref(), dt)
            * y.as_ref();
        t += dt;
        t_points.push(t);
        c0.push(y[(0, 0)]);
        c1.push(y[(1, 0)]);
        c2.push(y[(2, 0)]);
    }

    // plot...
    println!("c0: {:?}", &c0);
    #[cfg(feature = "plotters")]
    plot_time_series(t_points, c0, c1, c2);
}

#[cfg(feature = "plotters")]
fn plot_time_series(t: Vec<f64>, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("ex_linsys.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d((0f64..10000f64).log_scale(), (1e-9f64..2.0f64).log_scale())?;

    chart.configure_mesh()
        .y_desc("Population")
        .x_desc("Time")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
            t.clone().into_iter().zip(x),
            &RED,
        ))?
        .label("n0")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
            t.clone().into_iter().zip(y),
            &BLUE,
        ))?
        .label("n1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
            t.clone().into_iter().zip(z),
            &GREEN,
        ))?
        .label("n2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
