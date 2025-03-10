/// Bateman example
/// showing use of BDF, RK, and EPIRK time integrators
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
    let test_sys = TestBatemanFdSys::new();

    // initial species concentrations
    let y0 = faer::mat![
        [0.001,],
        [0.1,],
        [1.0,],
        ];

    // setup the integrator
    // let mut sys_solver = ode_bdf::BdfIntegrator::new(0.0, y0.as_ref(), 2, &test_sys);
    // let mut sys_solver = ode_rk::RkIntegrator::new(0.0, y0.as_ref(), 2, &test_sys);
    let iom = 2;
    let krylov_dim = 3;
    let expmv = Box::new(matexp_pade::PadeExpm::new(12));
    let matexp_m = matexp_krylov::KrylovExpm::new(expmv, krylov_dim, Some(iom));
    let mut sys_solver = ode_epirk::EpirkIntegrator::new(
        0.0, y0.as_ref(), "epi2".to_string(), &test_sys, matexp_m).with_opt(String::from("tol_fdt"), 1e-8);

    let mut t_points: Vec<f64> = Vec::new();
    // output concentrations
    let mut c0: Vec<f64> = Vec::new();
    let mut c1: Vec<f64> = Vec::new();
    let mut c2: Vec<f64> = Vec::new();

    // step the solution forward
    let mut t = 0.0;
    let dt = 5.0;
    let nsteps = 100;
    for _i in 0..nsteps {
        let y_new = sys_solver.step(dt).unwrap();

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

    // plot_time_series(t_points.clone(), c0.clone(), c1.clone(), c2.clone());
}


// fn plot_time_series(t: Vec<f64>, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
//     let root = BitMapBackend::new("ex_bateman.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE)?;
//     let mut chart = ChartBuilder::on(&root)
//         .margin(5)
//         .x_label_area_size(30)
//         .y_label_area_size(30)
//         .build_cartesian_2d((0f64..10000f64).log_scale(), (1e-9f64..2.0f64).log_scale())?;
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
//         .label("n0")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
// 
//     chart
//         .draw_series(LineSeries::new(
//             // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
//             t.clone().into_iter().zip(y),
//             &BLUE,
//         ))?
//         .label("n1")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
// 
//     chart
//         .draw_series(LineSeries::new(
//             // (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
//             t.clone().into_iter().zip(z),
//             &GREEN,
//         ))?
//         .label("n2")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
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
