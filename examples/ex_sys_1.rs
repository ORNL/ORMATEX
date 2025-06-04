/// Lotka voltera example showing use of BDF, RK, and EPIRK time integrators
use faer::prelude::*;
use ormatex::ode_sys::*;
use ormatex::ode_bdf;
use ormatex::ode_rk;
use ormatex::ode_epirk;
use ormatex::matexp_krylov;
use ormatex::ode_test_common::*;
use ormatex::matexp_pade;
// use plotters::prelude::*;


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
    // let mut sys_solver = ode_bdf::BdfIntegrator::new(0.0, y0.as_ref(), 2, &test_sys);
    // let mut sys_solver = ode_rk::RkIntegrator::new(0.0, y0.as_ref(), 2, &test_sys);
    let iom = 4;
    let krylov_dim = 4;
    let expmv = Box::new(matexp_pade::PadeExpm::new(12));
    let matexp_m = matexp_krylov::KrylovExpm::new(expmv, krylov_dim, Some(iom));
    let mut sys_solver = ode_epirk::EpirkIntegrator::new(
        0.0, y0.as_ref(), "epi3".to_string(), &test_sys, matexp_m);

    let mut t_points: Vec<f64> = Vec::new();
    let mut y_prey: Vec<f64> = Vec::new();
    let mut y_pred: Vec<f64> = Vec::new();

    // step the solution forward
    let mut t = 0.0;
    let dt = 0.1;
    let nsteps = 500;
    for _i in 0..nsteps {
        let y_new = sys_solver.step(dt).unwrap();

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

    // let plot_prefix: String = "ex_sys_".to_owned();
    // plot_time_series(t_points.clone(), y_prey.clone(), y_pred.clone(), plot_prefix);
}

// fn plot_time_series(t: Vec<f64>, x: Vec<f64>, y: Vec<f64>, mut prefix: String) -> Result<(), Box<dyn std::error::Error>> {
//     prefix.push_str("_ex_1.png");
//     let root = BitMapBackend::new(&prefix, (640, 480)).into_drawing_area();
//     root.fill(&WHITE)?;
//     let mut chart = ChartBuilder::on(&root)
//         .margin(5)
//         .x_label_area_size(30)
//         .y_label_area_size(30)
//         .build_cartesian_2d(-0f64..40f64, -0.1f64..9f64)?;
// 
//     chart.configure_mesh()
//         .y_desc("Population")
//         .x_desc("Time")
//         .draw()?;
// 
//     chart
//         .draw_series(LineSeries::new(
//             // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
//             t.clone().into_iter().zip(x),
//             &RED,
//         ))?
//         .label("Prey")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
// 
//     chart
//         .draw_series(LineSeries::new(
//             // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
//             t.clone().into_iter().zip(y),
//             &BLUE,
//         ))?
//         .label("Pred")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
// 
//     chart
//         .configure_series_labels()
//         .background_style(&WHITE.mix(0.8))
//         .border_style(&BLACK)
//         .draw()?;
// 
//     root.present()?;
// 
//     Ok(())
// }
