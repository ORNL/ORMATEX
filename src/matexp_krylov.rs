// Krylov Matrix Exponential Methods
//
use faer::prelude::*;
use faer::linop::LinOp;
use crate::arnoldi::arnoldi_lop;
use crate::matexp_pade;

/// Krylov methods to compute Sparse Matrix Exponential
/// and Phi functions
#[derive(Debug)]
pub struct KrylovExpm {
    /// max krylov dim size
    krylov_dim: usize,
    /// incomplete ortho depth
    iom: usize,
}

impl KrylovExpm {
    pub fn new(krylov_dim: usize, iom_in: Option<usize>) -> Self {
        assert!(krylov_dim > 0);
        Self {
            krylov_dim,
            iom: iom_in.unwrap_or(2),
        }
    }

    /// Computes exp(A*dt)*v0 when A is a linear operator
    pub fn apply_linop(&self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>)
        -> Mat<f64>
    {
        let (q, h, _b) = arnoldi_lop(a_lo, dt, v0.as_ref(), self.krylov_dim, self.iom);
        let matexp = matexp_pade::matexp(h.as_ref(), 1.0);
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(matexp.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        return faer::scale(beta) * (q.as_ref() * matexp.as_ref() * unit_vec)
    }

    /// Computes phi_k(A*dt) * v0 where A is a LinOp
    pub fn apply_phi_linop(
        &self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>, k: usize)
        -> Mat<f64>
    {
        let (q, h, _b) = arnoldi_lop(a_lo, 1.0, v0.as_ref(), self.krylov_dim, self.iom);
        let phi_k = matexp_pade::phi_ext((faer::scale(dt) * h.as_ref()).as_ref(), k);
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(phi_k.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        return faer::scale(beta) * (q.as_ref() * phi_k.as_ref() * unit_vec)
    }

    /// Computes tripplet (phi_k(A*dt)*v0, phi_k(A*dt*2)*v0, phi_k(A*dt*3)*v0)
    /// where A is a LinOp
    /// This saves two calls to arnoldi.
    ///
    /// From ref:  M. Hochbruck, C. Lubich and H. Selhofer.
    /// Exponential Integrators for Large
    /// Systems of Differential Equations.  J. Sci. Comp. 1996.
    pub fn apply_phi_linop_3(
        &self, a_lo: &dyn LinOp<f64>, dt: f64, v0: MatRef<f64>, k: usize)
        -> (Mat<f64>, Mat<f64>, Mat<f64>)
    {
        let (q, h, _b) = arnoldi_lop(a_lo, 1.0, v0.as_ref(), self.krylov_dim, self.iom);
        let phi_k = matexp_pade::phi_ext((faer::scale(dt) * h.as_ref()).as_ref(), k);
        let id = faer::Mat::<f64>::identity(phi_k.nrows(), phi_k.ncols());
        let beta = v0.norm_l2();
        let mut unit_vec = faer::Mat::zeros(phi_k.nrows(), 1);
        unit_vec[(0, 0)] = 1.0;
        // compute unscaled phi_k_1 = phi_k(A*dt)*v0
        let phi_k_1 = faer::scale(beta) * (q.as_ref() * phi_k.as_ref() * unit_vec.as_ref());
        // compute scaled phi_k_2 = phi_k(A*dt*2)*v0
        let phi_2tau_h = (faer::scale(dt * 1./2.) * h.as_ref() * phi_k.as_ref() + id.as_ref()) * phi_k.as_ref();
        let phi_k_2 = q.as_ref()
            * phi_2tau_h.as_ref()
            * unit_vec.as_ref() * faer::scale(beta);
        // compute scaled phi_k_3 = phi_k(A*dt*3)*v0
        let phi_k_3 = q.as_ref()
            * (faer::scale(2./3.) * (faer::scale(dt) * h.as_ref() * phi_k.as_ref() + id.as_ref())
            * phi_2tau_h.as_ref() + faer::scale(1./3.) * phi_k.as_ref())
            * unit_vec.as_ref() * faer::scale(beta);
        (phi_k_1, phi_k_2, phi_k_3)
    }
}


#[cfg(test)]
mod test_matexp_rs {
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

}
