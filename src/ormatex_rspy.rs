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

use numpy::{IntoPyArray, PyArray1, PyArray2,
            PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::{pymethods, pymodule, types::PyModule, PyResult, Python};
use pyo3::types::{PyList, PyDict};
use std::fmt;

use faer::prelude::*;
use faer_ext::*;
use faer::Par;
use faer::matrix_free::LinOp;
use faer::dyn_stack::PodStack;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};

use std::cell::RefCell;
use std::rc::Rc;
use reborrow::{ReborrowMut, Reborrow};

use crate::ode_sys::*;
use crate::ode_sys::*;
use crate::arnoldi::arnoldi_lop;
use crate::ode_bdf;
use crate::ode_rk;
use crate::ode_epirk;
use crate::matexp_krylov;
use crate::matexp_pade::{PadeExpm, DensePhikvEvaluator};
use crate::matexp_cauchy;

/// Wrapper around python PySys object
#[pyclass]
pub struct PySysWrapped {
    // alias of PyObject
    // pub py_sys: Py<PyAny>,
    pub py_sys: PyObject,
}

#[pymethods]
impl PySysWrapped {
    #[new]
    pub fn new(py_sys: PyObject) -> Self {
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
    py_linop: PyObject,
}

#[pymethods]
impl PyJaxJacLinOp {
    #[new]
    pub fn new(py_linop: PyObject) -> Self {
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
        // Not implented error!
        panic!("Not Implemented");
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
        let j_v = Python::with_gil(|py| {
            // convert MatRef to PyArray
            let x_slice = rhs.col(0).try_as_col_major().unwrap().as_slice();
            let x_np = x_slice.to_vec().into_pyarray(py);
            let j_v_py = self.py_linop.call_method(py, "matvec_npcompat", (x_np,), None).unwrap();
            let inner_bound = j_v_py.downcast_bound::<PyArray1<f64>>(py).unwrap();
            let inner: PyReadonlyArray1<f64> = inner_bound.extract().unwrap();
            let slice_view = inner.as_slice().unwrap();
            faer::col::ColRef::from_slice(slice_view).as_mat().to_owned()
        });

        out.copy_from(j_v);
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
        Python::with_gil(|py| {
            // convert x to numpy array
            let x_ndarray = x.into_ndarray().to_owned();
            let x_np = x_ndarray.into_pyarray(py);
            // rhs calc
            let frhs_x_py = self.py_sys.call_method(
                py, "frhs", (t, x_np), None).unwrap();
            // convert np result to faer mat
            let frhs_x_arr_bound = frhs_x_py.downcast_bound::<PyArray1<f64>>(py).unwrap();
            let inner: PyReadonlyArray1<f64> = frhs_x_arr_bound.extract().unwrap();
            let slice_view = inner.as_slice().unwrap();
            let frhs_x_mat = faer::col::ColRef::from_slice(slice_view).as_mat().to_owned();
            frhs_x_mat
        })
    }

    fn fjac<'b>(&'_ self,
                t: f64,
                x: MatRef<'b, f64>)
            -> Box<dyn LinOp<f64> + '_> {
        // Box::new(get_fd_jac(self, t, x))
        Python::with_gil(|py| {
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
fn select_solver<'a>(
    sys: &'a PySysWrapped,
    t0: f64,
    y0_mat: MatRef<'_, f64>,
    method: String,
    expmv_method: String,
    krylov_dim: usize,
    iom: usize,
    tol_fdt: f64,
    )
    -> Rc < RefCell<dyn IntegrateSys<'a, TimeType=f64, SysStateType=Mat<f64>> + 'a> >
{
    if method.as_str() == "bdf1" || method.as_str() == "backeuler" {
        return Rc::new( RefCell::new(ode_bdf::BdfIntegrator::new(t0, y0_mat, 1, sys)))
    }
    else if method.as_str() == "bdf2" {
        return Rc::new( RefCell::new(ode_bdf::BdfIntegrator::new(t0, y0_mat, 2, sys)))
    }
    else if method.as_str() == "cn" {
        return Rc::new( RefCell::new(ode_bdf::BdfIntegrator::new(t0, y0_mat, 3, sys)))
    }
    // exp integrator family is default
    let expmv: Box<dyn DensePhikvEvaluator> = match expmv_method.as_str() {
        "cram" => { Box::new(matexp_cauchy::gen_cram_expm(16)) },
        "parabolic" => { Box::new(matexp_cauchy::gen_parabolic_expm(24)) },
        // pade is default
        _ => { Box::new(PadeExpm::new(12)) },
    };
    let matexp_m = matexp_krylov::KrylovExpm::new(expmv, krylov_dim, Some(iom));
    Rc::new( RefCell::new(ode_epirk::EpirkIntegrator::new(
        t0, y0_mat, method, sys, matexp_m).with_opt(String::from("tol_fdt"), tol_fdt)))
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
    let kd = kwds.unwrap_or(PyDict::new(py));
    let method: String = kd.as_ref().get_item("method").and_then(|item| item.extract::<String>()).unwrap_or(String::from("epi2"));
    let expmv_method: String = kd.as_ref().get_item("expmv_method").and_then(|item| item.extract::<String>()).unwrap_or(String::from("pade"));
    let krylov_dim: usize = kd.as_ref().get_item("max_krylov_dim").and_then(|item| item.extract::<usize>()).unwrap_or(100);
    let iom: usize = kd.as_ref().get_item("iom").and_then(|item| item.extract::<usize>()).unwrap_or(2);
    let tol: f64 = kd.as_ref().get_item("tol").and_then(|item| item.extract::<f64>()).unwrap_or(1e-8);
    let tol_fdt: f64 = kd.as_ref().get_item("tol_fdt").and_then(|item| item.extract::<f64>()).unwrap_or(1e-6);
    let osteps: usize = kd.as_ref().get_item("osteps").and_then(|item| item.extract::<usize>()).unwrap_or(1);

    let y = y0.as_array();
    let y0_mat = y.view().into_faer();

    // setup the integrator
    let solver = select_solver(
        sys, t0, y0_mat, method, expmv_method, krylov_dim, iom, tol_fdt);

    // storage for results
    let mut y_out: Vec<Bound<PyArray2<f64>>> = Vec::with_capacity(nsteps);
    let mut t_out: Vec<f64> = Vec::with_capacity(nsteps);

    // integrate the sys
    let mut borrowed_solver = solver.borrow_mut();
    for i in 0..nsteps {
        if i % osteps == 0 {
            let _y = borrowed_solver.state();
            let _t = borrowed_solver.time();
            y_out.push(_y.as_ref().into_ndarray().to_owned().into_pyarray(py));
            t_out.push(_t);
        }
        let y_new = borrowed_solver.step(dt);
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
    py_linop: PyObject,
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
    let b_ndarray = b.as_array();
    let b_mat = b_ndarray.view().into_faer();

    // run arnoldi
    let (q, h, bkdwn) = arnoldi_lop(
        &lop_wrapped, a_lo_scale, b_mat.as_ref(), m, iom);

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
            "cram" => { Box::new(matexp_cauchy::gen_cram_expm(order)) },
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
        let a_arr = a_np.as_array();
        let a = a_arr.view().into_faer();
        let v0_arr = v0_np.as_array();
        let v0 = v0_arr.view().into_faer();
        let phikv = self.evaluator.phik_apply(a, dt, v0, k);
        let ndarray_phikv = phikv.as_ref().into_ndarray().to_owned();
        ndarray_phikv.into_pyarray(py).to_owned().into()
    }
}

#[pymodule]
#[pyo3(name="ormatex")]
fn ormatex<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()>
{
    // Adds PySys wrapper
    m.add_class::<PySysWrapped>()?;

    // Adds dense phi_k(A*dt)*v0 evaluator
    m.add_class::<DensePhikvEvalRs>()?;

    // Adds rust ormatex integrate method
    m.add_function(wrap_pyfunction!(integrate_wrapper_rs, m)?)?;

    // Adds rust based arnoldi method
    m.add_function(wrap_pyfunction!(arnoldi_rs, m)?)?;

    Ok(())
}
