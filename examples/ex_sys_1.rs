/// Lotka voltera example showing use of BDF, RK, and EPIRK time integrators
/// Run example with plot:
/// cargo run --example ex_sys_1 --features plot
use faer::prelude::*;
use ormatex::ode_sys::*;
use ormatex::ode_bdf;
use ormatex::ode_rk;
use ormatex::ode_epirk;
use ormatex::matexp_krylov;
use ormatex::matexp_leja;
use ormatex::test_common::*;
use ormatex::matexp_pade;
use ormatex::logger::init_logger;

// optional deps for plotting
#[cfg(feature="plot")]
use kuva::prelude::*;


pub fn main() {
    // setup system
    let test_sys = TestLvFdSys::new();
    // let test_sys = TestLvSys::new();

    // initial conds
    let y0 = faer::mat![
        [5.0,], // pred pop
        [4.0,], // prey pop
        ];

    // setup the integrator
    // let mut sys_solver = ode_bdf::BdfIntegrator::new(0.0, y0.as_ref(), 2);
    // let mut sys_solver = ode_rk::RkIntegrator::new(0.0, y0.as_ref(), 2);
    // let iom = 2;
    // let krylov_dim = 4;
    // let expmv = Box::new(matexp_pade::PadeExpm::new(12));
    // let mut matexp_m = matexp_krylov::KrylovExpm::new(expmv, krylov_dim, Some(iom));

    let _logger = init_logger();
    let lp = matexp_leja::LejaPoints::new_from_lib("leja_circle").slice(0, 80);
    let leja_ellipse_adapter = matexp_leja::LejaEllipseAdapterArnoldiIOM::new(
        -1.0, 0.0, 1.0, 1e-18, 20, 2, 1.1);
    let matexp_m = matexp_leja::LejaPhiEval::new(
        lp, 20, 1e-12, "clapm", "dd_phi", false, Box::new(leja_ellipse_adapter));

    let mut sys_solver = ode_epirk::EpirkIntegrator::<matexp_leja::LejaPhiEval>::new(
        0.0, y0.as_ref(), "epi3".to_string(), matexp_m);

    let mut t_points: Vec<f64> = Vec::new();
    let mut y_prey: Vec<f64> = Vec::new();
    let mut y_pred: Vec<f64> = Vec::new();

    // step the solution forward
    let mut t = 0.0;
    let dt = 0.1;
    let nsteps = 100;
    for _i in 0..nsteps {
        let y_new = sys_solver.step(&test_sys, dt).unwrap();

        t_points.push(t);
        y_prey.push((&y_new).y[(0, 0)]);
        y_pred.push((&y_new).y[(1, 0)]);

        sys_solver.accept_step(y_new);
        t += dt;
    }

    // print the results
    println!("t, pred, prey");
    for i in 0..nsteps {
        println!("{:?}, {:?}, {:?}", t_points[i], y_pred[i], y_prey[i]);
    }

    #[cfg(feature="plot")]
    plot_time_series(t_points.clone(), y_prey.clone(), y_pred.clone());
}

#[cfg(feature="plot")]
fn plot_time_series(t: Vec<f64>, y0: Vec<f64>, y1: Vec<f64>)
    -> Result<(), Box<dyn std::error::Error>>
{
    // create plots
    let plots = vec![
        Plot::Line(LinePlot::new()
            // iter() yeilds &T and into_iter yeilds T
            .with_data(t.clone().into_iter().zip(y0.clone().into_iter()))
            .with_color("steelblue")
            .with_legend("y0")
        ),
        Plot::Line(LinePlot::new()
            .with_data(t.clone().into_iter().zip(y1.clone().into_iter()))
            .with_color("crimson")
            .with_legend("y1")
        ),
    ];
    let layout = Layout::auto_from_plots(&plots)
        .with_x_label("x")
        .with_y_label("y");
    let svg = render_to_svg(plots, layout);
    std::fs::write("ex_1.svg", svg)?;

    Ok(())
}
