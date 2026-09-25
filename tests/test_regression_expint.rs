use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_leja::{
    LejaEllipseAdapterArnoldiIOM, LejaEllipseAdapterStatic, LejaPhiEval, LejaPoints,
};
use ormatex::matexp_pade::PadeExpm;
use ormatex::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_exprb::ExprbIntegrator;
use ormatex::ode_sys::OdeSys;
use ormatex::ode_traits::IntegrateSys;
use ormatex::test_common::TestBatemanFdSys;

#[derive(Clone, Copy)]
enum EvaluatorKind {
    Krylov,
    LejaStatic,
    LejaAdaptive,
}

struct ScalarNonautoSys;

impl<'a> OdeSys<'a> for ScalarNonautoSys {
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
        faer::mat![[-t * x[(0, 0)]]]
    }

    fn fjac<'b>(&'a self, t: f64, _x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(faer::mat![[-t]])
    }
}

fn krylov_evaluator() -> KrylovExpm {
    KrylovExpm::new(Box::new(PadeExpm::new(12)), 6, 80, 1e-12, Some(2))
}

fn leja_evaluator(adaptive: bool, a: f64, b: f64, c: f64) -> LejaPhiEval {
    let points = LejaPoints::new_from_lib("leja_circle").slice(0, 160);
    let adapter: Box<dyn ormatex::matexp_leja::GetSpectrumBounds> = if adaptive {
        Box::new(LejaEllipseAdapterArnoldiIOM::new(
            a, b, c, -1.0, 10, 2, 1.05,
        ))
    } else {
        Box::new(LejaEllipseAdapterStatic::new(a, b, c))
    };
    LejaPhiEval::new(
        points,
        150,
        1e-12,
        "clapm",
        "dd_taylor",
        false,
        adapter,
    )
}

fn run_steps<'a>(
    solver: &mut dyn IntegrateSys<'a, TimeType = f64, SysStateType = Mat<f64>>,
    sys: &'a dyn OdeSys<'a>,
    dt: f64,
    nsteps: usize,
) {
    for _ in 0..nsteps {
        let result = solver.step(sys, dt).unwrap();
        solver.accept_step(result);
    }
}

fn assert_state_close(label: &str, state: &Mat<f64>, reference: &Mat<f64>, tol_rel: f64) {
    let error = (state.as_ref() - reference.as_ref()).norm_max();
    let scale = reference.norm_max().max(1e-12);
    let relative_error = error / scale;
    assert!(
        relative_error <= tol_rel,
        "{label}: state={state:?}, reference={reference:?}, rel-err={relative_error:.3e}, tol={tol_rel:.3e}"
    );
}

fn bateman_reference(t: f64, y0: MatRef<f64>) -> Mat<f64> {
    let lambda_0 = 1.0e-3;
    let lambda_1 = 1.0e1;
    let lambda_2 = 1.0e-1;
    let bateman_matrix = faer::mat![
        [-lambda_0, lambda_1, 0.0],
        [0.0, -lambda_1, lambda_2],
        [0.0, 0.0, -lambda_2],
    ];
    PadeExpm::new(12).apply_phi_k(bateman_matrix.as_ref(), t, y0, 0)
}

fn run_bateman<T: LinOpPhikvEvaluator>(method: &str, expm: T) -> Mat<f64> {
    let sys = TestBatemanFdSys::new();
    let y0 = faer::mat![[0.001_f64], [0.1_f64], [1.0_f64]];
    let mut solver = match method {
        "exprb3" => {
            let mut solver = ExprbIntegrator::new(0.0, y0.as_ref(), method.to_string(), expm);
            run_steps(&mut solver, &sys, 0.5, 20);
            return solver.state();
        }
        "epi3" => EpirkIntegrator::new(0.0, y0.as_ref(), method.to_string(), expm),
        _ => panic!("unsupported exponential method: {method}"),
    };
    run_steps(&mut solver, &sys, 0.5, 20);
    solver.state()
}

fn run_nonauto<T: LinOpPhikvEvaluator>(method: &str, expm: T) -> Mat<f64> {
    let sys = ScalarNonautoSys;
    let y0 = faer::mat![[1.0_f64]];
    let dt = 0.05;
    let nsteps = 20;
    let mut solver = match method {
        "exprb3" => {
            let mut solver = ExprbIntegrator::new(0.0, y0.as_ref(), method.to_string(), expm)
                .with_opt("tol_fdt".to_string(), 1e-12);
            for _ in 0..nsteps {
                let result = solver.step(&sys, dt).unwrap();
                assert!(result.err.is_some_and(f64::is_finite));
                solver.accept_step(result);
            }
            return solver.state();
        }
        "epi3" => EpirkIntegrator::new(0.0, y0.as_ref(), method.to_string(), expm)
            .with_opt("tol_fdt".to_string(), 1e-12),
        _ => panic!("unsupported exponential method: {method}"),
    };
    run_steps(&mut solver, &sys, dt, nsteps);
    solver.state()
}

fn run_bateman_case(method: &str, evaluator: EvaluatorKind) -> Mat<f64> {
    match evaluator {
        EvaluatorKind::Krylov => run_bateman(method, krylov_evaluator()),
        EvaluatorKind::LejaStatic => {
            run_bateman(method, leja_evaluator(false, -10.0, 0.0, 0.0))
        }
        EvaluatorKind::LejaAdaptive => {
            run_bateman(method, leja_evaluator(true, -10.0, 0.0, 0.0))
        }
    }
}

fn run_nonauto_case(method: &str, evaluator: EvaluatorKind) -> Mat<f64> {
    match evaluator {
        EvaluatorKind::Krylov => run_nonauto(method, krylov_evaluator()),
        EvaluatorKind::LejaStatic => {
            run_nonauto(method, leja_evaluator(false, -1.0, 0.0, 0.0))
        }
        EvaluatorKind::LejaAdaptive => {
            run_nonauto(method, leja_evaluator(true, -1.0, 0.0, 0.0))
        }
    }
}

#[test]
fn expint_regression_bateman3() {
    let y0 = faer::mat![[0.001_f64], [0.1_f64], [1.0_f64]];
    let reference = bateman_reference(10.0, y0.as_ref());
    let cases = [
        ("exprb3", EvaluatorKind::Krylov, "EXPRB3/Krylov"),
        ("exprb3", EvaluatorKind::LejaStatic, "EXPRB3/LejaStatic"),
        ("exprb3", EvaluatorKind::LejaAdaptive, "EXPRB3/LejaAdaptive"),
        ("epi3", EvaluatorKind::Krylov, "EPI3/Krylov"),
        ("epi3", EvaluatorKind::LejaStatic, "EPI3/LejaStatic"),
        ("epi3", EvaluatorKind::LejaAdaptive, "EPI3/LejaAdaptive"),
    ];

    for (method, evaluator, label) in cases {
        let state = run_bateman_case(method, evaluator);
        assert_state_close(label, &state, &reference, 1e-7);
    }
}

#[test]
fn expint_regression_nonauto() {
    let reference = faer::mat![[( -0.5_f64).exp()]];
    let cases = [
        ("exprb3", EvaluatorKind::Krylov, "EXPRB3/Krylov"),
        ("exprb3", EvaluatorKind::LejaStatic, "EXPRB3/LejaStatic"),
        ("exprb3", EvaluatorKind::LejaAdaptive, "EXPRB3/LejaAdaptive"),
        ("epi3", EvaluatorKind::Krylov, "EPI3/Krylov"),
        ("epi3", EvaluatorKind::LejaStatic, "EPI3/LejaStatic"),
        ("epi3", EvaluatorKind::LejaAdaptive, "EPI3/LejaAdaptive"),
    ];

    for (method, evaluator, label) in cases {
        let state = run_nonauto_case(method, evaluator);
        assert_state_close(label, &state, &reference, 1e-3);
    }
}
