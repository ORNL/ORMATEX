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

use numpy::ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2,
    PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2,
    PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyRuntimeError, pymethods, pymodule, types::PyModule, PyResult, Python};
use pyo3::types::{PyList, PyDict};
use pyo3::prelude::*;

use faer::prelude::*;
use faer_ext::*;

use faer::linop::LinOp;
use crate::ode_sys::*;

use crate::ode_sys::*;
use crate::ode_bdf;
use crate::ode_rk;
use crate::ode_epirk;
use crate::matexp_krylov;

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
            let frhs_x_mat = faer::col::from_slice(slice_view).as_2d().to_owned();
            frhs_x_mat
        })
    }

    fn fjac<'b>(&'_ self,
                t: f64,
                x: MatRef<'b, f64>)
            -> Box<dyn LinOp<f64> + '_> {
        Box::new(get_fd_jac(self, t, x))
    }
}

#[pyfunction]
fn integrate_wrapper_rs<'py>(
    py: Python<'py>,
    sys: &PySysWrapped,
    y0: PyReadonlyArray2<f64>,
    t0: f64,
    dt: f64,
    nsteps: usize,
    krylov_dim: usize,
    )
    -> (Bound<'py, PyList>, Bound<'py, PyList>)
{
    let y = y0.as_array();
    let y0_mat = y.view().into_faer();

    // setup the integrator
    let iom = 2;
    let order = 3;
    let matexp_m = matexp_krylov::KrylovExpm::new(krylov_dim, Some(iom));
    let mut sys_solver = ode_epirk::EpirkIntegrator::new(
        t0, y0_mat.as_ref(), order, sys, matexp_m);

    // storage for results
    let mut y_out: Vec<Bound<PyArray2<f64>>> = Vec::with_capacity(nsteps);
    let mut t_out: Vec<f64> = Vec::with_capacity(nsteps);

    // integrate the sys
    for _i in 0..nsteps {
        let y_new = sys_solver.step(dt).unwrap();
        sys_solver.accept_step(y_new);
        let _y = sys_solver.state();
        let _t = sys_solver.time();
        y_out.push(_y.as_ref().into_ndarray().to_owned().into_pyarray(py));
        t_out.push(_t.clone());
    }
    let y_out_pylist = PyList::new(py, y_out).unwrap();
    let t_out_pylist = PyList::new(py, t_out).unwrap();

    (y_out_pylist, t_out_pylist)
}

#[pymodule]
#[pyo3(name="ormatex")]
fn ormatex<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()>
{
    // Adds PySys wrapper
    m.add_class::<PySysWrapped>()?;

    // Adds rust ormatex integrate method
    let _ = m.add_function(wrap_pyfunction!(integrate_wrapper_rs, m)?);

    Ok(())
}
