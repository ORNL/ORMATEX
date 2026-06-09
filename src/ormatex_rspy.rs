/*
 * Copyright© 2025 UT-Battelle, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/// Python interface to Rust ormatex integrators
///
/// See readme for python module install and use.
///
/// Wraps a python ODE Sys object to be compatible
/// with the Rust based ormatex integrators.
/// This interface allows interoperability between
/// numpy/jax backed ODE models with Rust based
/// temporal integration procedures.  The primary benifit is
/// the ability to use JAX-based AD methods to compute
/// system jacobian and jabobian-vector products while
/// also leveraging rust-based dense and sparse linear algebra
/// routines for performant time integration method implementations
/// on the CPU.
///
use pyo3::prelude::*;
use pyo3::{pymethods, pymodule, Python};
use pyo3::types::{PyList, PyDict};
use numpy::{IntoPyArray, PyArray1, PyArray2,
            PyReadonlyArray1, PyReadonlyArray2};
use flexi_logger::{LoggerHandle};

use faer::prelude::*;
use faer_ext::*;
use faer::Par;
use faer::matrix_free::LinOp;
use faer::dyn_stack::{MemStack, StackReq};

use std::fmt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ode_sys::*;
use crate::logger::init_logger;
use crate::ode_implicit;
use crate::tableau_implicit::ImplicitBT;
use crate::ode_rk;
use crate::ode_epirk;
use crate::matexp_krylov;
use crate::matexp_leja;
use crate::matexp_cauchy;
use crate::matexp_pade::{PadeExpm, phi_ext};
use crate::matexp_traits::{DensePhikvEvaluator, LinOpPhikvEvaluator};
use crate::arnoldi::arnoldi_lop;


/// Wrapper around python PySys object
#[pyclass]
pub struct PySysWrapped {
    // alias of PyObject
    pub py_sys: Py<PyAny>,
}

#[pymethods]
impl PySysWrapped {
    #[new]
    pub fn new(py_sys: Py<PyAny>) -> Self {
        // let gil = Python::acquire_gil();
        Self {
            py_sys,
        }
    }
}

/// LinOp for python JAX-based linear operator
#[pyclass]
pub struct PyJaxJacLinOp {
    /// inner linop def in python
    /// see omatex_py.ode_sys.LinOp for def
    py_linop: Py<PyAny>,
}

#[pymethods]
impl PyJaxJacLinOp {
    #[new]
    pub fn new(py_linop: Py<PyAny>) -> Self {
        // let gil = Python::acquire_gil();
        Self {
            py_linop,
        }
    }
}
impl fmt::Debug for PyJaxJacLinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Py LinOp x={:?} \n ", self.py_linop)
    }
}
impl LinOp<f64> for PyJaxJacLinOp {
    fn apply_scratch(
            &self,
            rhs_ncols: usize,
            parallelism: Par,
        ) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        StackReq::empty()
    }

    /// Number of rows in the linop
    fn nrows(&self) -> usize {
        let nr: usize = Python::attach(|py| {
            let dim_py = self.py_linop.call_method(py, "dim", (), None).unwrap();
            let inner_bound = dim_py.downcast_bound(py).unwrap();
            let inner: usize = inner_bound.extract().unwrap();
            inner
        });
        nr
    }

    /// Number of cols in the linop
    fn ncols(&self) -> usize {
        // Not implented error!
        panic!("Not Implemented");
    }

    fn apply(
        &self,
        mut out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        // unused
        _ = parallelism;
        _ = stack;

        // compute jacobian vector product in python
        Python::attach(|py| {
            // convert MatRef to PyArray
            let x_slice = rhs.col(0).try_as_col_major().unwrap().as_slice();
            let x_np = x_slice.to_vec().into_pyarray(py);
            let j_v_py = self.py_linop.call_method(py, "matvec_npcompat", (x_np,), None).unwrap();
            let inner_bound = j_v_py.downcast_bound::<PyArray1<f64>>(py).unwrap();
            let inner: PyReadonlyArray1<f64> = inner_bound.extract().unwrap();
            out.col_mut(0).copy_from(inner.into_faer());
        });
    }

    fn conj_apply(
            &self,
            out: MatMut<'_, f64>,
            rhs: MatRef<'_, f64>,
            parallelism: Par,
            stack: &mut MemStack,
        ) {
        // Not implented error!
        panic!("Not Implemented");
    }
}


/// Implement required OdeSys interface for interop
/// with Rust ormatex integrators.  Calls the
/// python implementations via pyO3 obj.call_method()
impl OdeSys<'_> for PySysWrapped {
    fn frhs(&self, t: f64, x: MatRef<f64>) -> Mat<f64> {
        Python::attach(|py| {
            // convert x to numpy array
            let x_ndarray = x.into_ndarray().to_owned();
            let x_np = x_ndarray.into_pyarray(py);
            // rhs calc
            let frhs_x_py = self.py_sys.call_method(
                py, "frhs", (t, x_np), None).unwrap();
            // convert np result to faer mat
            let frhs_x_arr_bound = frhs_x_py.downcast_bound::<PyArray1<f64>>(py).unwrap();
            let inner: PyReadonlyArray1<f64> = frhs_x_arr_bound.extract().unwrap();
            inner.into_faer().as_mat().to_owned()
        })
    }

    fn fjac<'b>(&'_ self,
                t: f64,
                x: MatRef<'b, f64>)
            -> Box<dyn LinOp<f64> + '_> {
        // Box::new(get_fd_jac(self, t, x))
        Python::attach(|py| {
            // convert x to numpy array
            let x_ndarray = x.into_ndarray().to_owned();
            let x_np = x_ndarray.into_pyarray(py);
            // py based jacobian linop
            let fjac_py = self.py_sys.call_method(
                py, "fjac", (t, x_np), None).unwrap();
            // wrapped jacobian linop
            let fjac_inner = PyJaxJacLinOp::new(fjac_py);
            Box::new(fjac_inner)
        })
    }
}

/// Select ode solver
fn select_solver<'a, T: LinOpPhikvEvaluator + 'a>(
    t0: f64,
    y0_mat: MatRef<'_, f64>,
    method: String,
    tol_fdt: f64,
    tol_lin: f64,
    tol_nlin: f64,
    matexp_m: T,
    )
    -> Rc < RefCell<dyn IntegrateSys<'a, TimeType=f64, SysStateType=Mat<f64>> + 'a> >
{
    // backward euler
    if method.as_str() == "bdf1" || method.as_str() == "backeuler" {
        return Rc::new( RefCell::new(ode_implicit::BdfIntegrator::new(t0, y0_mat, 1, tol_lin, tol_nlin)))
    }
    // backward difference formula 2
    else if method.as_str() == "bdf2" {
        return Rc::new( RefCell::new(ode_implicit::BdfIntegrator::new(t0, y0_mat, 2, tol_lin, tol_nlin)))
    }
    // crank-nicolson
    else if method.as_str() == "cn" {
        return Rc::new(RefCell::new(
                ode_implicit::DirkIntegrator::new(t0, y0_mat, ImplicitBT::crank_nicolson(), tol_lin, tol_nlin)
                ))
    }
    // sdirk32
    else if method.as_str() == "sdirk32" {
        return Rc::new(RefCell::new(
                ode_implicit::DirkIntegrator::new(t0, y0_mat, ImplicitBT::sdirk32(), tol_lin, tol_nlin)
                ))
    }
    // sdirk33
    else if method.as_str() == "sdirk33" {
        return Rc::new(RefCell::new(
                ode_implicit::DirkIntegrator::new(t0, y0_mat, ImplicitBT::sdirk33(), tol_lin, tol_nlin)
                ))
    }
    // forward euler
    else if method.as_str() == "rk1" || method.as_str() == "forwardeuler" {
        return Rc::new( RefCell::new(ode_rk::RkIntegrator::new(t0, y0_mat, 1)))
    }
    // rk4
    else if method.as_str() == "rk4" {
        return Rc::new( RefCell::new(ode_rk::RkIntegrator::new(t0, y0_mat, 4)))
    }
    // exponential integrator fallthrough
    Rc::new( RefCell::new(ode_epirk::EpirkIntegrator::new(
        t0, y0_mat, method, matexp_m).with_opt(String::from("tol_fdt"), tol_fdt)))
}


fn get_val_or_default<'py, T>(py: Python<'py>, kd_hash: &HashMap<String, Py<PyAny>>, key: String, default: T) -> T
where T: FromPyObject<'py>
{
    for (k, v) in kd_hash.iter() {
        if *k == key {
            return v.extract(py).unwrap_or(default);
        }
    }
    default
}

#[pyfunction]
#[pyo3(signature = (sys, y0, t0, dt, nsteps, **kwds))]
fn integrate_wrapper_rs<'py>(
    py: Python<'py>,
    sys: &PySysWrapped,
    y0: PyReadonlyArray2<f64>,
    t0: f64,
    dt: f64,
    nsteps: usize,
    kwds: Option<Bound<'py, PyDict>>
    )
    -> (Bound<'py, PyList>, Bound<'py, PyList>)
{
    // process kwargs
    let kd: pyo3::Bound<'_, PyDict> = kwds.unwrap_or(PyDict::new(py));
    let kd_hash: HashMap<String, Py<PyAny>> = kd.extract().unwrap_or(HashMap::new());

    // stepper settings
    let method: String = get_val_or_default(py, &kd_hash, String::from("method"), String::from("epi2"));
    let phikv_method: String = get_val_or_default(py, &kd_hash, String::from("phikv_method"), String::from("krylov"));
    let expmv_method: String = get_val_or_default(py, &kd_hash, String::from("expmv_method"), String::from("pade"));
    let max_krylov_dim: usize = get_val_or_default(py, &kd_hash, String::from("max_krylov_dim"), 100);
    let m: usize = get_val_or_default(py, &kd_hash, String::from("m"), max_krylov_dim);
    let iom: usize = get_val_or_default(py, &kd_hash, String::from("iom"), 2);
    let max_substeps: usize = get_val_or_default(py, &kd_hash, String::from("max_substeps"), 0);
    let tol: f64 = get_val_or_default(py, &kd_hash, String::from("tol"), 1e-8);
    let tol_fdt: f64 = get_val_or_default(py, &kd_hash, String::from("tol_fdt"), 1e-8);
    let osteps: usize = get_val_or_default(py, &kd_hash, String::from("osteps"), 1);
    // linear and nonlinear solver settings
    let tol_lin: f64 = get_val_or_default(py, &kd_hash, String::from("tol_lin"), 1e-8);
    let tol_nlin: f64 = get_val_or_default(py, &kd_hash, String::from("tol_nlin"), 1e-8);
    // jacobian spectrum analysis settings
    let leja_a: f64 = get_val_or_default(py, &kd_hash, String::from("leja_a"), -1.0);
    let leja_b: f64 = get_val_or_default(py, &kd_hash, String::from("leja_b"), 0.0);
    let leja_c: f64 = get_val_or_default(py, &kd_hash, String::from("leja_c"), 1.0);
    let spec_tol: f64 = get_val_or_default(py, &kd_hash, String::from("spec_tol"), 1.0e-8);
    let spec_iter: usize = get_val_or_default(py, &kd_hash, String::from("spec_iter"), 20);
    let spec_method: String = get_val_or_default(py, &kd_hash, String::from("spec_method"), String::from("arnoldi"));
    let dd_method: String = get_val_or_default(py, &kd_hash, String::from("dd_method"), String::from("dd_phi"));
    let krylov_reuse: bool = get_val_or_default(py, &kd_hash, String::from("krylov_reuse"), false);
    // optional logging settings
    let logging: bool = get_val_or_default(py, &kd_hash, String::from("logging"), false);
    let _logger: Option<LoggerHandle> = if logging {Some(init_logger())} else { None };

    let y0_mat = y0.into_faer();

    // setup the dense phi evaluator
    let expmv: Box<dyn DensePhikvEvaluator> = match expmv_method.as_str() {
        "cram" | "cram_16" => { Box::new(matexp_cauchy::gen_cram_expm(16)) },
        "parabolic" => { Box::new(matexp_cauchy::gen_parabolic_expm(24)) },
        // pade is default
        _ => { Box::new(PadeExpm::new(12)) },
    };

    // setup the time integrator
    let solver = match phikv_method.as_str() {
        "leja" => {
            let lp = matexp_leja::LejaPoints::new_from_fn("leja_circle").slice(0, m+2);
            let mut matexp_m = match spec_method.as_str() {
                "none" => {
                    // user specified spectrum parameters
                    let leja_ellipse_adapter =
                        matexp_leja::LejaEllipseAdapterStatic::new(
                        leja_a, leja_b, leja_c);
                    // adaptive specturm parameter updates
                    matexp_leja::LejaPhiEval::new(
                        lp, std::cmp::min(m, 800), tol, "clapm", dd_method.as_str(),
                        krylov_reuse, Box::new(leja_ellipse_adapter))
                },
                _ => {
                    let leja_ellipse_adapter =
                        matexp_leja::LejaEllipseAdapterArnoldiIOM::new(
                        leja_a, leja_b, leja_c, spec_tol, spec_iter, iom, 1.0);
                    // adaptive specturm parameter updates
                    matexp_leja::LejaPhiEval::new(
                        lp, std::cmp::min(m, 800), tol, "clapm", dd_method.as_str(),
                        krylov_reuse, Box::new(leja_ellipse_adapter))
                }
            };
            matexp_m.set_max_substeps(max_substeps);
            select_solver(t0, y0_mat, method, tol_fdt, tol_lin, tol_nlin, matexp_m)
        },
        "taylor" => {
            let lp = matexp_leja::LejaPoints::new(vec![0.0; m], vec![0.0; m]);
            let leja_ellipse_adapter =
                matexp_leja::LejaEllipseAdapterStatic::new(
                leja_a, leja_b, leja_c);
            let mut matexp_m =
                matexp_leja::LejaPhiEval::new(
                    lp, std::cmp::min(m, 800), tol, "taylor", dd_method.as_str(),
                    krylov_reuse, Box::new(leja_ellipse_adapter));
            select_solver(t0, y0_mat, method, tol_fdt, tol_lin, tol_nlin, matexp_m)
        },
        // krylov is default
        _ => {
            let mut matexp_m = matexp_krylov::KrylovExpm::new(expmv, std::cmp::min(100, m), m, tol, Some(iom));
            select_solver(t0, y0_mat, method, tol_fdt, tol_lin, tol_nlin, matexp_m)
        },
    };

    // storage for results
    let mut y_out: Vec<Bound<PyArray2<f64>>> = Vec::with_capacity(nsteps);
    let mut t_out: Vec<f64> = Vec::with_capacity(nsteps);

    // integrate the sys
    let mut borrowed_solver = solver.borrow_mut();
    for i in 0..nsteps {
        if i % osteps == 0 || i == nsteps-1 {
            let _y = borrowed_solver.state();
            let _t = borrowed_solver.time();
            y_out.push(_y.as_ref().into_ndarray().to_owned().into_pyarray(py));
            t_out.push(_t);
        }
        let y_new = borrowed_solver.step(sys, dt);
        borrowed_solver.accept_step(y_new.unwrap());
    }
    let _y = borrowed_solver.state();
    let _t = borrowed_solver.time();
    y_out.push(_y.as_ref().into_ndarray().to_owned().into_pyarray(py));
    t_out.push(_t);
    let y_out_pylist = PyList::new(py, y_out).unwrap();
    let t_out_pylist = PyList::new(py, t_out).unwrap();

    (y_out_pylist, t_out_pylist)
}


/// Rust phi_k(A)
/// Note: phi_0(A) == exp(A)
#[pyfunction]
fn phi_k_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    k: usize,
)
    -> Bound<'py, PyArray2<f64>>
{
    // convert a mat into fear mat
    let a_mat = a.into_faer();

    // run phi_k(dt*A)
    let phik = phi_ext(a_mat, k);

    // convert faer mats into numpy arrays
    let phik_ndarray = phik.as_ref().into_ndarray().to_owned();
    phik_ndarray.into_pyarray(py)
}

/// Rust Arnoldi method binding for interop with python
///
/// * `py_linop` - python LinOp
/// * `b` - numpy vector
/// * `m` - max krylov iteration
/// * `iom` - incomplete ortho depth
///
/// returns
/// * `H` - Upper Hessenberge
/// * `V` - orthonormal basis
/// * `bkdwn` - iter where happy breakdown occured
///
#[pyfunction]
fn arnoldi_rs<'py>(
    py: Python<'py>,
    py_linop: Py<PyAny>,
    a_lo_scale: f64,
    b: PyReadonlyArray2<f64>,
    m: usize,
    iom: usize,
    )
    -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, usize)
{
    // create wrapper around python linop
    let lop_wrapped = PyJaxJacLinOp::new(py_linop);

    // convert b vec into fear mat
    let b_mat = b.into_faer();

    // run arnoldi
    let (q, h, bkdwn) = arnoldi_lop(
        &lop_wrapped, a_lo_scale, b_mat, m, iom);

    // convert faer mats into numpy arrays
    let h_ndarray = h.as_ref().into_ndarray().to_owned();
    let q_ndarray = q.as_ref().into_ndarray().to_owned();
    (
        q_ndarray.into_pyarray(py),
        h_ndarray.into_pyarray(py),
        bkdwn
    )
}


/// Python interface for computing dense phi_k(A*dt)*v0 products
#[pyclass(unsendable)]
pub struct DensePhikvEvalRs {
    method: String,
    order: usize,
    evaluator: Box<dyn DensePhikvEvaluator>
}

#[pymethods]
impl DensePhikvEvalRs {
    #[new]
    pub fn new(method: String, order: usize) -> Self {
        let evaluator: Box<dyn DensePhikvEvaluator> = match method.as_str() {
            "cram" | "cram_16" => { Box::new(matexp_cauchy::gen_cram_expm(order)) },
            "parabolic" => { Box::new(matexp_cauchy::gen_parabolic_expm(order)) },
            // pade is default
            _ => { Box::new(PadeExpm::new(order)) },
        };
        Self {
            method,
            order,
            evaluator
        }
    }

    pub fn eval(&self, py: Python<'_>, a_np: PyReadonlyArray2<f64>, dt: f64, v0_np: PyReadonlyArray2<f64>, k: usize)
        -> Py<PyArray2<f64>>
    {
        let a = a_np.into_faer();
        let v0 = v0_np.into_faer();
        let phikv = self.evaluator.phik_apply(a, dt, v0, k);
        let ndarray_phikv = phikv.as_ref().into_ndarray().to_owned();
        ndarray_phikv.into_pyarray(py).to_owned().into()
    }
}


#[pymodule(name="ormatex")]
mod ormatex {
    #[pymodule_export]
    use super::PySysWrapped;

    #[pymodule_export]
    use super::DensePhikvEvalRs;

    #[pymodule_export]
    use super::integrate_wrapper_rs;

    #[pymodule_export]
    use super::arnoldi_rs;

    #[pymodule_export]
    use super::phi_k_rs;
}
