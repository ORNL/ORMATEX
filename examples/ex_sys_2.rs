/// Bateman example
/// showing use of BDF, RK, and EPIRK time integrators
use faer::prelude::*;
use ormatex::ode_sys::*;
use ormatex::ode_implicit;
use ormatex::ode_rk;
use ormatex::ode_epirk;
use ormatex::matexp_krylov;
use ormatex::test_common::*;
use ormatex::matexp_pade;

// optional deps for plotting
#[cfg(feature="plot")]
use kuva::prelude::*;


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
    // let mut sys_solver = ode_implicit::BdfIntegrator::new(0.0, y0.as_ref(), 2);
    // let mut sys_solver = ode_rk::RkIntegrator::new(0.0, y0.as_ref(), 2);
    let iom = 2;
    let krylov_dim = 3;
    let m = 3;
    let tol = 1e-12;
    let expmv = Box::new(matexp_pade::PadeExpm::new(12));
    let mut matexp_m = matexp_krylov::KrylovExpm::new(expmv, m, krylov_dim, tol, Some(iom));
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
    // create plots
    let plots = vec![
        Plot::Line(LinePlot::new()
            // iter() yeilds &T and into_iter yeilds T
            .with_data(t.clone().into_iter().zip(c0.clone().into_iter()))
            .with_color("steelblue")
            .with_legend("c0")
        ),
        Plot::Line(LinePlot::new()
            .with_data(t.clone().into_iter().zip(c1.clone().into_iter()))
            .with_color("crimson")
            .with_legend("c1")
        ),
        Plot::Line(LinePlot::new()
            .with_data(t.clone().into_iter().zip(c2.clone().into_iter()))
            .with_color("seagreen")
            .with_legend("c2")
        ),
    ];
    let layout = Layout::auto_from_plots(&plots)
        .with_log_y()
        .with_log_x()
        .with_x_axis_min(0.1)
        .with_y_axis_min(1.0e-8)
        .with_x_label("x")
        .with_y_label("c");
    let svg = render_to_svg(plots, layout);
    std::fs::write("ex_2.svg", svg)?;

    Ok(())
}
