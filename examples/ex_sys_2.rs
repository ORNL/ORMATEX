/// Bateman example
/// showing use of BDF, RK, and EPIRK time integrators
use faer::prelude::*;
use ormatex::ode_sys::*;
use ormatex::ode_bdf;
use ormatex::ode_rk;
use ormatex::ode_epirk;
use ormatex::matexp_krylov;
use ormatex::test_common::*;
use ormatex::matexp_pade;

// optional deps for plotting
#[cfg(feature="plot")]
use plotlars::{ScatterPlot, LinePlot, Plot, Rgb};
#[cfg(feature="plot")]
use polars::prelude::*;


pub fn main() {
    // setup system
    let test_sys = TestBatemanFdSys::new();

    // initial species concentrations
    let y0 = faer::mat![
        [0.001,],
        [0.1,],
        [1.0,],
        ];

    // setup the integrator
    // let mut sys_solver = ode_bdf::BdfIntegrator::new(0.0, y0.as_ref(), 2);
    // let mut sys_solver = ode_rk::RkIntegrator::new(0.0, y0.as_ref(), 2);
    let iom = 2;
    let krylov_dim = 3;
    let expmv = Box::new(matexp_pade::PadeExpm::new(12));
    let matexp_m = matexp_krylov::KrylovExpm::new(expmv, krylov_dim, Some(iom));
    let mut sys_solver = ode_epirk::EpirkIntegrator::<matexp_krylov::KrylovExpm>::new(
        0.0, y0.as_ref(), "epi2".to_string(), matexp_m).with_opt(String::from("tol_fdt"), 1e-8);

    // output concentrations
    let mut t_points: Vec<f64> = Vec::new();
    let mut c0: Vec<f64> = Vec::new();
    let mut c1: Vec<f64> = Vec::new();
    let mut c2: Vec<f64> = Vec::new();

    // step the solution forward
    let mut t = 0.0;
    let dt = 5.0;
    let nsteps = 100;
    for _i in 0..nsteps {
        let y_new = sys_solver.step(&test_sys, dt).unwrap();

        t_points.push(t);
        c0.push((&y_new).y[(0, 0)]);
        c1.push((&y_new).y[(1, 0)]);
        c2.push((&y_new).y[(2, 0)]);

        sys_solver.accept_step(y_new);
        t += dt;

    }

    // print the results
    println!("t, x0, x1, x2");
    for i in 0..nsteps {
        println!("{:?}, {:?}, {:?}, {:?}", t_points[i], c0[i], c1[i], c2[i]);
    }

    #[cfg(feature="plot")]
    plot_time_series(t_points.clone(), c0.clone(), c1.clone(), c2.clone());
}

#[cfg(feature="plot")]
fn plot_time_series(t: Vec<f64>, c0: Vec<f64>, c1: Vec<f64>, c2: Vec<f64>)
    -> Result<(), Box<dyn std::error::Error>>
{
    // create polars dataframe from vecs
    let df = df! [
        "t" => t,
        "y0" => c0,
        "y1" => c1,
        "y2" => c2,
    ]?;

    // plot the dataframe contents
    let plot = LinePlot::builder()
        .data(&df)
        .x("t")
        .y("y0")
        .additional_lines(vec!["y1", "y2"])
        .build();
    plot.write_image("ex_sys_2.png", 1200, 800, 2.0)?;

    Ok(())
}
