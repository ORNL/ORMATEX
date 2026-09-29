/*
 * Copyright(c) 2025 UT-Battelle, LLC
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
//! Python (pyo3) bindings for the Rust ORMATEX integrators.
//!
//! Built with the `python` feature and imported in python as
//! `ormatex_py.ormatex`. See the readme for module install and use. The
//! high level python entry point is `ormatex_py.integrate_wrapper.integrate`
//! with a method name ending in `_rs` (for example `method="epi2_rs"`), which
//! calls `integrate_wrapper_rs` here.
//!
//! [`PySysWrapped`] wraps a python ODE system object so that it implements
//! the Rust [`OdeSys`] trait. This allows numpy/JAX backed ODE models to be
//! advanced by the Rust time integrators. The primary benefit is the ability
//! to use JAX-based automatic differentiation to compute the system Jacobian
//! and Jacobian-vector products, while leveraging Rust dense and sparse linear
//! algebra for performant time integration on the CPU. Also exposed are dense
//! phi-function evaluation ([`DensePhikvEvalRs`], `phi_k_rs`), an Arnoldi
//! routine (`arnoldi_rs`) and Leja point evaluators for diagonal matrices.
use flexi_logger::LoggerHandle;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};
use pyo3::{pymethods, pymodule, Python};

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::Par;
use faer_ext::*;

use std::collections::HashMap;
use std::fmt;

use crate::arnoldi::arnoldi_lop;
use crate::logger::init_logger;
use crate::matexp_cauchy;
use crate::matexp_leja::{complex_diag_leja_phikv_fitted, complex_diag_leja_phikv_static};
use crate::matexp_pade::{phi_ext, PadeExpm};
use crate::matexp_traits::DensePhikvEvaluator;
use crate::integrator_builder::{
    DenseExpmMethod, ExponentialEvaluator, ExponentialIntegratorBuilder, ExponentialMethod,
    ExplicitIntegratorBuilder, ExplicitMethod, ImplicitIntegratorBuilder, ImplicitMethod,
    KrylovOptions, LejaDdMethod, LejaOptions, LejaSpectrum, TaylorOptions,
};
use crate::ode_sys::*;

use std::str::FromStr;

/// Wrapper around a python ODE system object.
///
/// The wrapped python object must provide `frhs(t, x)`, returning the rhs as a
/// 1D numpy array, and `fjac(t, x)`, returning a linear operator object (see
/// [`PyJaxJacLinOp`]). In python, create it with `PySysWrapped(ode_sys)`, see
/// `ormatex_py.ode_sys.OdeSysNp`.
#[pyclass]
pub struct PySysWrapped {
    /// The wrapped python system object (alias of PyObject).
    pub py_sys: Py<PyAny>,
}

#[pymethods]
impl PySysWrapped {
    /// Wrap a python ODE system object.
    #[new]
    pub fn new(py_sys: Py<PyAny>) -> Self {
        // let gil = Python::acquire_gil();
        Self { py_sys }
    }
}

/// Rust `LinOp` wrapping a python (JAX-based) Jacobian linear operator.
///
/// The python object must provide `dim()`, the operator dimension, and
/// `matvec_npcompat(x)`, the matrix-vector product on a 1D numpy array. Only
/// square operators and `apply` are supported; `ncols` and `conj_apply` panic.
#[pyclass]
pub struct PyJaxJacLinOp {
    /// inner linop def in python
    /// see omatex_py.ode_sys.LinOp for def
    py_linop: Py<PyAny>,
}

#[pymethods]
impl PyJaxJacLinOp {
    /// Wrap a python linear operator object.
    #[new]
    pub fn new(py_linop: Py<PyAny>) -> Self {
        // let gil = Python::acquire_gil();
        Self { py_linop }
    }
}
impl fmt::Debug for PyJaxJacLinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Py LinOp x={:?} \n ", self.py_linop)
    }
}
impl LinOp<f64> for PyJaxJacLinOp {
    fn apply_scratch(&self, rhs_ncols: usize, parallelism: Par) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        StackReq::empty()
    }

    /// Number of rows in the linop
    fn nrows(&self) -> usize {
        let nr: usize = Python::attach(|py| {
            let dim_py = self.py_linop.call_method(py, "dim", (), None).unwrap();
            let inner_bound = dim_py.cast_bound(py).unwrap();
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
        out: MatMut<f64>,
        rhs: MatRef<f64>,
        parallelism: Par,
        stack: &mut MemStack,
    ) {
        // unused
        _ = parallelism;
        _ = stack;

        // compute jacobian vector product in python
        Python::attach(|py| {
            // convert MatRef to PyArray
            let x_slice = rhs.col(0).try_as_col_major().unwrap().as_slice();
            let x_np = x_slice.to_vec().into_pyarray(py);
            let j_v_py = self
                .py_linop
                .call_method(py, "matvec_npcompat", (x_np,), None)
                .unwrap();
            let inner_bound = j_v_py.cast_bound::<PyArray1<f64>>(py).unwrap();
            let inner: PyReadonlyArray1<f64> = inner_bound.extract().unwrap();
            out.col_mut(0).copy_from(inner.into_faer());
        });
    }

    fn conj_apply(
        &self,
        _out: MatMut<'_, f64>,
        _rhs: MatRef<'_, f64>,
        _parallelism: Par,
        _stack: &mut MemStack,
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
            let frhs_x_py = self
                .py_sys
                .call_method(py, "frhs", (t, x_np), None)
                .unwrap();
            // convert np result to faer mat
            let frhs_x_arr_bound = frhs_x_py.cast_bound::<PyArray1<f64>>(py).unwrap();
            let inner: PyReadonlyArray1<f64> = frhs_x_arr_bound.extract().unwrap();
            inner.into_faer().as_mat().to_owned()
        })
    }

    fn fjac<'b>(&'_ self, t: f64, x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + '_> {
        // Box::new(get_fd_jac(self, t, x))
        Python::attach(|py| {
            // convert x to numpy array
            let x_ndarray = x.into_ndarray().to_owned();
            let x_np = x_ndarray.into_pyarray(py);
            // py based jacobian linop
            let fjac_py = self
                .py_sys
                .call_method(py, "fjac", (t, x_np), None)
                .unwrap();
            // wrapped jacobian linop
            let fjac_inner = PyJaxJacLinOp::new(fjac_py);
            Box::new(fjac_inner)
        })
    }
}

/// Look up `key` in the keyword dict, returning `default` if the key is missing
/// or its value cannot be converted to `T`.
fn get_val_or_default<'a, 'py, T>(
    py: Python<'py>,
    kd_hash: &'a HashMap<String, Py<PyAny>>,
    key: String,
    default: T,
) -> T
where
    T: FromPyObject<'a, 'py>,
{
    for (k, v) in kd_hash.iter() {
        if *k == key {
            return v.extract(py).unwrap_or(default);
        }
    }
    default
}

/// Integrate an ODE system with a Rust integrator built from keyword options.
///
/// Called from python by `ormatex_py.integrate_wrapper.integrate` for methods
/// with an `_rs` suffix. Returns `(y_list, t_list)`: lists of numpy state
/// arrays and times. Outputs are the initial state, the state before every
/// `osteps`-th step and before the last step, and the final state.
///
/// Arguments:
///
/// * `sys` - the wrapped python ODE system, `PySysWrapped`
/// * `y0` - initial state, 2D numpy array of shape `(n, 1)`
/// * `t0` - initial time
/// * `dt` - fixed time step size
/// * `nsteps` - number of steps
/// * `**kwds` - options, all optional (defaults in parentheses). A value that
///   cannot be converted to the expected type silently falls back to its
///   default.
///
/// Options:
///
/// * `method` - integration method name (`epi2`), one of the names accepted by
///   `ExplicitMethod`, `ImplicitMethod` or `ExponentialMethod` (case
///   insensitive), for example `rk4`, `bdf2`, `cn`, `sdirk33`, `epi3`, `exprb3`
/// * `osteps` - output every `osteps` steps (1); must be greater than 0
/// * `logging` - write a log file with `init_logger` (False)
/// * `tol_lin`, `tol_nlin` - implicit method linear and Newton tolerances (1e-8)
/// * `tol_fdt` - exponential methods, nonautonomous correction threshold (1e-8)
/// * `phi_method` - exponential methods, evaluator: `krylov`, `leja` or `taylor` (`krylov`)
/// * `tol` - evaluator tolerance (1e-8)
/// * `m` - Krylov initial dimension or Leja/Taylor polynomial degree (`max_krylov_dim`)
/// * `max_krylov_dim` - maximum Krylov dimension (100)
/// * `iom` - incomplete orthogonalization depth (2)
/// * `expmv_method` - Krylov dense method: `pade`, `cram`, `cram_16`, `parabolic` (`pade`)
/// * `dd_method` - Leja/Taylor divided differences: `dd_phi` or `dd_taylor` (`dd_phi`)
/// * `krylov_reuse` - Leja/Taylor, re-use Krylov information (False)
/// * `max_substeps` - Leja, number of substeps, 0 for none (0)
/// * `leja_a`, `leja_b`, `leja_c` - Leja/Taylor spectrum bounds (-1.0, 0.0, 1.0)
/// * `spec_method` - Leja spectrum, `arnoldi` (adaptive) or `none` (static) (`arnoldi`)
/// * `spec_tol`, `spec_iter`, `spec_saftey_factor` - adaptive Leja spectrum
///   options (1e-8, 20, 1.05)
///
/// Raises `ValueError` for an unknown method or option value, an invalid
/// configuration, or a failed time step.
#[pyfunction]
#[pyo3(signature = (sys, y0, t0, dt, nsteps, **kwds))]
fn integrate_wrapper_rs<'py>(
    py: Python<'py>,
    sys: &PySysWrapped,
    y0: PyReadonlyArray2<f64>,
    t0: f64,
    dt: f64,
    nsteps: usize,
    kwds: Option<Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    // process kwargs
    let kd: pyo3::Bound<'_, PyDict> = kwds.unwrap_or(PyDict::new(py));
    let kd_hash: HashMap<String, Py<PyAny>> = kd.extract().unwrap_or(HashMap::new());

    // integrator method settings
    let method: String =
        get_val_or_default(py, &kd_hash, String::from("method"), String::from("epi2"));
    let phi_method: String =
        get_val_or_default(py, &kd_hash, String::from("phi_method"), String::from("krylov"));
    let expmv_method: String =
        get_val_or_default(py, &kd_hash, String::from("expmv_method"), String::from("pade"));
    let tol_fdt: f64 =
        get_val_or_default(py, &kd_hash, String::from("tol_fdt"), 1e-8);
    let tol: f64 =
        get_val_or_default(py, &kd_hash, String::from("tol"), 1e-8);
    let tol_lin: f64 =
        get_val_or_default(py, &kd_hash, String::from("tol_lin"), 1e-8);
    let tol_nlin: f64 =
        get_val_or_default(py, &kd_hash, String::from("tol_nlin"), 1e-8);
    let m_default: usize =
        get_val_or_default(py, &kd_hash, String::from("max_krylov_dim"), 100);
    let m: usize =
        get_val_or_default(py, &kd_hash, String::from("m"), m_default);
    let iom: usize =
        get_val_or_default(py, &kd_hash, String::from("iom"), 2);
    let max_krylov_dim: usize =
        get_val_or_default(py, &kd_hash, String::from("max_krylov_dim"), 100);
    let leja_a: f64 =
        get_val_or_default(py, &kd_hash, String::from("leja_a"), -1.0);
    let leja_b: f64 =
        get_val_or_default(py, &kd_hash, String::from("leja_b"), 0.0);
    let leja_c: f64 =
        get_val_or_default(py, &kd_hash, String::from("leja_c"), 1.0);
    let dd_method: String =
        get_val_or_default(py, &kd_hash, String::from("dd_method"), String::from("dd_phi"));
    let krylov_reuse: bool =
        get_val_or_default(py, &kd_hash, String::from("krylov_reuse"), false);
    let max_substeps: usize =
        get_val_or_default(py, &kd_hash, String::from("max_substeps"), 0);
    let spec_tol: f64 =
        get_val_or_default(py, &kd_hash, String::from("spec_tol"), 1.0e-8);
    let spec_iter: usize =
        get_val_or_default(py, &kd_hash, String::from("spec_iter"), 20);
    let spec_method: String =
        get_val_or_default(py,&kd_hash,String::from("spec_method"),String::from("arnoldi"));
    let safety_factor: f64 =
        get_val_or_default(py,&kd_hash,String::from("spec_saftey_factor"),1.05);
    let osteps: usize =
        get_val_or_default(py, &kd_hash, String::from("osteps"), 1);
    if osteps == 0 {
        return Err(PyValueError::new_err("osteps must be >0"));
    }
    // optional logging settings
    let logging: bool = get_val_or_default(py, &kd_hash, String::from("logging"), false);
    let _logger: Option<LoggerHandle> = if logging { Some(init_logger()) } else { None };

    // initial solution state
    let y0_mat = y0.into_faer();

    // setup the solver
    let solver = if let Ok(explicit_method) = ExplicitMethod::from_str(&method) {
        ExplicitIntegratorBuilder::new(t0, y0_mat, explicit_method)
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    } else if let Ok(implicit_method) = ImplicitMethod::from_str(&method) {
        ImplicitIntegratorBuilder::new(t0, y0_mat, implicit_method)
            .with_tol_lin(tol_lin)
            .with_tol_nlin(tol_nlin)
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    } else if let Ok(exponential_method) = ExponentialMethod::from_str(&method) {
        let evaluator = match phi_method.to_ascii_lowercase().as_str() {
            "krylov" => {
                let dense_method = DenseExpmMethod::from_str(&expmv_method)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                ExponentialEvaluator::Krylov(
                    KrylovOptions::default()
                        .with_dense_method(dense_method)
                        .with_m(m)
                        .with_max_dim(max_krylov_dim)
                        .with_iom(iom)
                        .with_tol(tol),
                )
            }
            "leja" => {
                let spectrum = if spec_method.eq_ignore_ascii_case("none") {
                    LejaSpectrum::static_bounds(leja_a, leja_b, leja_c)
                } else if spec_method.eq_ignore_ascii_case("arnoldi") {
                    LejaSpectrum::adaptive(
                        leja_a,
                        leja_b,
                        leja_c,
                        spec_tol,
                        spec_iter,
                        iom,
                        safety_factor,
                    )
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unsupported Leja spectrum method: {spec_method}"
                    )));
                };
                let dd_method = LejaDdMethod::from_str(&dd_method)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                ExponentialEvaluator::Leja(
                    LejaOptions::default()
                        .with_m(m)
                        .with_max_substeps(max_substeps)
                        .with_tol(tol)
                        .with_dd_method(dd_method)
                        .with_krylov_reuse(krylov_reuse)
                        .with_spectrum(spectrum),
                )
            }
            "taylor" => {
                let dd_method = LejaDdMethod::from_str(&dd_method)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                ExponentialEvaluator::Taylor(
                    TaylorOptions::default()
                        .with_m(m)
                        .with_tol(tol)
                        .with_dd_method(dd_method)
                        .with_krylov_reuse(krylov_reuse)
                        .with_bounds(leja_a, leja_b, leja_c),
                )
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported phi evaluation method: {phi_method}"
                )));
            }
        };
        ExponentialIntegratorBuilder::new(t0, y0_mat, exponential_method)
            .with_tol_fdt(tol_fdt)
            .with_evaluator(evaluator)
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    } else {
        return Err(PyValueError::new_err(format!(
            "unsupported time integration method: {method}"
        )));
    };

    // storage for results
    let mut y_out: Vec<Bound<PyArray2<f64>>> = Vec::with_capacity(nsteps);
    let mut t_out: Vec<f64> = Vec::with_capacity(nsteps);

    // integrate the sys
    let mut solver = solver;
    for i in 0..nsteps {
        if i % osteps == 0 || i == nsteps - 1 {
            let _y = solver.state();
            let _t = solver.time();
            y_out.push(_y.as_ref().into_ndarray().to_owned().into_pyarray(py));
            t_out.push(_t);
        }
        let y_new = solver
            .step(sys, dt)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        solver.accept_step(y_new);
    }
    let _y = solver.state();
    let _t = solver.time();
    y_out.push(_y.as_ref().into_ndarray().to_owned().into_pyarray(py));
    t_out.push(_t);
    let y_out_pylist = PyList::new(py, y_out).unwrap();
    let t_out_pylist = PyList::new(py, t_out).unwrap();

    Ok((y_out_pylist, t_out_pylist))
}

/// Rust dense $\varphi_k(A)$ matrix function, with $\varphi_0(A) = \exp(A)$.
///
/// Arguments: `a` - 2D numpy array (square matrix), `k` - phi-function order.
/// Returns the 2D numpy array $\varphi_k(A)$. Any time step scaling must be
/// applied to `a` by the caller.
#[pyfunction]
fn phi_k_rs<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>, k: usize) -> Bound<'py, PyArray2<f64>> {
    // convert a mat into fear mat
    let a_mat = a.into_faer();

    // run phi_k(dt*A)
    let phik = phi_ext(a_mat, k);

    // convert faer mats into numpy arrays
    let phik_ndarray = phik.as_ref().into_ndarray().to_owned();
    phik_ndarray.into_pyarray(py)
}

/// Rust Arnoldi method binding for interop with python.
///
/// # Arguments
///
/// * `py_linop` - python LinOp (see `ormatex_py.ode_sys.LinOp`)
/// * `a_lo_scale` - scale factor applied to the linear operator
/// * `b` - starting vector, 2D numpy array of shape `(n, 1)`
/// * `m` - max krylov iteration
/// * `iom` - incomplete ortho depth
///
/// # Returns
///
/// A tuple `(V, H, bkdwn)`:
///
/// * `V` - orthonormal basis
/// * `H` - upper Hessenberg matrix
/// * `bkdwn` - iteration count reached (less than `m` on happy breakdown)
#[pyfunction]
fn arnoldi_rs<'py>(
    py: Python<'py>,
    py_linop: Py<PyAny>,
    a_lo_scale: f64,
    b: PyReadonlyArray2<f64>,
    m: usize,
    iom: usize,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, usize) {
    // create wrapper around python linop
    let lop_wrapped = PyJaxJacLinOp::new(py_linop);

    // convert b vec into fear mat
    let b_mat = b.into_faer();

    // run arnoldi
    let (q, h, bkdwn) = arnoldi_lop(&lop_wrapped, a_lo_scale, b_mat, m, iom);

    // convert faer mats into numpy arrays
    let h_ndarray = h.as_ref().into_ndarray().to_owned();
    let q_ndarray = q.as_ref().into_ndarray().to_owned();
    (
        q_ndarray.into_pyarray(py),
        h_ndarray.into_pyarray(py),
        bkdwn,
    )
}

/// Leja point evaluation of $\varphi_k(\Delta t D) v$ for a diagonal complex matrix $D$ with static spectrum bounds.
///
/// Wraps [`crate::matexp_leja::complex_diag_leja_phikv_static`]; see it for details.
///
/// # Arguments
///
/// * `a`, `b`, `c` - spectrum bounds: min real part, max real part, max imaginary magnitude
/// * `dt` - time step scale
/// * `d_diag_re`, `d_diag_im` - real and imaginary parts of the diagonal of $D$
/// * `v_re`, `v_im` - real and imaginary parts of the vector $v$
/// * `k` - phi-function order, 0 for the matrix exponential
/// * `m` - maximum Leja polynomial degree
///
/// # Returns
///
/// A tuple `(phikv_re, phikv_im, leja_re, leja_im)` of real and imaginary
/// parts of the result and of the scaled Leja points used.
#[pyfunction]
fn complex_diag_leja_phikv_static_rs<'py>(
    py: Python<'py>,
    a: f64,
    b: f64,
    c: f64,
    dt: f64,
    d_diag_re: PyReadonlyArray1<f64>,
    d_diag_im: PyReadonlyArray1<f64>,
    v_re: PyReadonlyArray1<f64>,
    v_im: PyReadonlyArray1<f64>,
    k: usize,
    m: usize,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    // convert vecs into fear col
    let v_re_col = v_re.into_faer();
    let v_im_col = v_im.into_faer();
    let d_diag_re_col = d_diag_re.into_faer();
    let d_diag_im_col = d_diag_im.into_faer();

    let (phikv_re, phikv_im, lp_sc_re, lp_sc_im) = complex_diag_leja_phikv_static(
        a,
        b,
        c,
        dt,
        d_diag_re_col,
        d_diag_im_col,
        v_re_col,
        v_im_col,
        k,
        m,
    );

    // convert output to numpy
    let phikv_re_ndarray = phikv_re.as_mat().as_ref().into_ndarray().to_owned();
    let phikv_im_ndarray = phikv_im.as_mat().as_ref().into_ndarray().to_owned();
    let lp_sc_re_ndarray = lp_sc_re.as_mat().as_ref().into_ndarray().to_owned();
    let lp_sc_im_ndarray = lp_sc_im.as_mat().as_ref().into_ndarray().to_owned();
    (
        phikv_re_ndarray.into_pyarray(py),
        phikv_im_ndarray.into_pyarray(py),
        lp_sc_re_ndarray.into_pyarray(py),
        lp_sc_im_ndarray.into_pyarray(py),
    )
}

/// Leja point evaluation of $\varphi_k(\Delta t D) v$ for a diagonal complex matrix $D$ with spectrum bounds fitted by Arnoldi iteration.
///
/// Wraps [`crate::matexp_leja::complex_diag_leja_phikv_fitted`]; see it for details.
///
/// # Arguments
///
/// * `a` - dense matrix (2D numpy array) whose spectrum is fitted
/// * `b` - starting vector for the Arnoldi iteration, 2D numpy array `(n, 1)`
/// * `dt` - time step scale
/// * `d_diag_re`, `d_diag_im` - real and imaginary parts of the diagonal of $D$
/// * `v_re`, `v_im` - real and imaginary parts of the vector $v$
/// * `k` - phi-function order, 0 for the matrix exponential
/// * `m` - maximum Leja polynomial degree
/// * `iom` - incomplete orthogonalization depth
/// * `n_ritz` - number of Arnoldi iterations used in the spectrum fit
/// * `krylov_reuse` - re-use Krylov information in the Leja interpolation
/// * `spec_saftey_factor` - scale factor applied to the fitted spectrum bounds
///
/// # Returns
///
/// A tuple `(phikv_re, phikv_im, leja_re, leja_im)` of real and imaginary
/// parts of the result and of the scaled Leja points used.
#[pyfunction]
fn complex_diag_leja_phikv_fitted_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    dt: f64,
    d_diag_re: PyReadonlyArray1<f64>,
    d_diag_im: PyReadonlyArray1<f64>,
    v_re: PyReadonlyArray1<f64>,
    v_im: PyReadonlyArray1<f64>,
    k: usize,
    m: usize,
    iom: usize,
    n_ritz: usize,
    krylov_reuse: bool,
    spec_saftey_factor: f64,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
) {
    // create wrapper around python linop
    let a_mat = a.into_faer();

    // convert b vec into fear mat
    let b_mat = b.into_faer();

    // convert vecs into fear col
    let v_re_col = v_re.into_faer();
    let v_im_col = v_im.into_faer();
    let d_diag_re_col = d_diag_re.into_faer();
    let d_diag_im_col = d_diag_im.into_faer();

    let (phikv_re, phikv_im, lp_sc_re, lp_sc_im) = complex_diag_leja_phikv_fitted(
        &a_mat,
        b_mat.as_ref(),
        dt,
        d_diag_re_col,
        d_diag_im_col,
        v_re_col,
        v_im_col,
        k,
        m,
        iom,
        n_ritz,
        krylov_reuse,
        Some(spec_saftey_factor),
    );

    // convert output to numpy
    let phikv_re_ndarray = phikv_re.as_mat().as_ref().into_ndarray().to_owned();
    let phikv_im_ndarray = phikv_im.as_mat().as_ref().into_ndarray().to_owned();
    let lp_sc_re_ndarray = lp_sc_re.as_mat().as_ref().into_ndarray().to_owned();
    let lp_sc_im_ndarray = lp_sc_im.as_mat().as_ref().into_ndarray().to_owned();
    (
        phikv_re_ndarray.into_pyarray(py),
        phikv_im_ndarray.into_pyarray(py),
        lp_sc_re_ndarray.into_pyarray(py),
        lp_sc_im_ndarray.into_pyarray(py),
    )
}

/// Python interface for computing dense $\varphi_k(A \Delta t) v_0$ products.
///
/// Wraps a [`DensePhikvEvaluator`]. Used in python by
/// `ormatex_py.phi_evaluator`.
#[pyclass(unsendable)]
pub struct DensePhikvEvalRs {
    _method: String,
    _order: usize,
    evaluator: Box<dyn DensePhikvEvaluator>,
}

#[pymethods]
impl DensePhikvEvalRs {
    /// Create a dense phi-function evaluator.
    ///
    /// # Arguments
    ///
    /// * `method` - `cram` or `cram_16` (CRAM, only order 16 is supported),
    ///   `parabolic` (parabolic contour, even order), any other value selects Pade
    /// * `order` - order of the method; for Pade the maximum number of squarings
    #[new]
    pub fn new(method: String, order: usize) -> Self {
        let evaluator: Box<dyn DensePhikvEvaluator> = match method.as_str() {
            "cram" | "cram_16" => Box::new(matexp_cauchy::gen_cram_expm(order)),
            "parabolic" => Box::new(matexp_cauchy::gen_parabolic_expm(order)),
            // pade is default
            _ => Box::new(PadeExpm::new(order)),
        };
        Self {
            _method: method,
            _order: order,
            evaluator,
        }
    }

    /// Precompute data (for example factorizations) for the given matrix and step.
    ///
    /// # Arguments
    ///
    /// * `a_np` - 2D numpy array, the matrix $A$
    /// * `dt` - time step scale
    /// * `v0_np` - 2D numpy array, the vector
    /// * `k` - phi-function order
    pub fn prepare(
        &mut self,
        _py: Python<'_>,
        a_np: PyReadonlyArray2<f64>,
        dt: f64,
        v0_np: PyReadonlyArray2<f64>,
        k: usize,
    ) {
        let a = a_np.into_faer();
        let v0 = v0_np.into_faer();
        self.evaluator.apply_prepare(a, dt, v0, k);
    }

    /// Evaluate $\varphi_k(\Delta t A) v_0$.
    ///
    /// # Arguments
    ///
    /// * `a_np` - 2D numpy array, the matrix $A$
    /// * `dt` - time step scale
    /// * `v0_np` - 2D numpy array, the vector $v_0$ of shape `(n, 1)`
    /// * `k` - phi-function order
    ///
    /// # Returns
    ///
    /// The 2D numpy array $\varphi_k(\Delta t A) v_0$.
    pub fn eval_phik(
        &self,
        py: Python<'_>,
        a_np: PyReadonlyArray2<f64>,
        dt: f64,
        v0_np: PyReadonlyArray2<f64>,
        k: usize,
    ) -> Py<PyArray2<f64>> {
        let a = a_np.into_faer();
        let v0 = v0_np.into_faer();
        let phikv = self.evaluator.apply_phi_k(a, dt, v0, k);
        let ndarray_phikv = phikv.as_ref().into_ndarray().to_owned();
        ndarray_phikv.into_pyarray(py).to_owned().into()
    }

    /// Evaluate the sum $\sum_j \varphi_{k_j}(\Delta t A) b_j$.
    ///
    /// # Arguments
    ///
    /// * `a_np` - 2D numpy array, the matrix $A$
    /// * `dt` - time step scale
    /// * `bs_np` - list of 2D numpy arrays $b_j$ of shape `(n, 1)`
    /// * `ks` - list of phi-function orders $k_j$, one per vector
    ///
    /// # Returns
    ///
    /// The 2D numpy array holding the summed product.
    pub fn eval_phik_v(
        &self, py: Python<'_>,
        a_np: PyReadonlyArray2<f64>,
        dt: f64,
        bs_np: Vec<PyReadonlyArray2<f64>>,
        ks: Vec<usize>
    ) -> Py<PyArray2<f64>>
    {
        let a = a_np.into_faer();
        let bs: Vec<MatRef<f64>> = bs_np.iter().map(|b| b.clone().into_faer()).collect();
        let phikv = self.evaluator.apply_phi_k_v(a, dt, &bs, &ks);
        let ndarray_phikv = phikv.as_ref().into_ndarray().to_owned();
        ndarray_phikv.into_pyarray(py).to_owned().into()
    }
}

/// ORMATEX Rust bindings: integrators and phi-function evaluators.
#[pymodule(name = "ormatex")]
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
    use super::complex_diag_leja_phikv_static_rs;

    #[pymodule_export]
    use super::complex_diag_leja_phikv_fitted_rs;

    #[pymodule_export]
    use super::phi_k_rs;
}
