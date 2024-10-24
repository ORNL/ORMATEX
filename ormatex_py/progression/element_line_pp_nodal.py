import logging
from typing import Type

import numpy as np
from numpy.polynomial.legendre import Legendre

from skfem.element.element_h1 import ElementH1
from skfem.refdom import Refdom, RefLine

from scipy.interpolate import BarycentricInterpolator

logger = logging.getLogger(__name__)

class GLL_quad:
    def __init__(self):
        self.GL_quad = {
            2: (np.array([[-1., 1.]]), np.array([1., 1.])),
            3: (np.array([[-1., 0., 1]]), np.array([1./3, 4./3, 1./3])),
            4: (np.array([[-1., -1./np.sqrt(5.), 1./np.sqrt(5.), 1.]]), np.array([1./6, 5./6, 5./6, 1./6])),
            5: (np.array([[-1., -np.sqrt(3./7), 0., np.sqrt(3./7), 1.]]), np.array([1./10, 49./90, 32./45, 49./90, 1./10]))
        }
        for k, (X, W) in self.GL_quad.items():
            # transform to reference element
            Xt = (1+X)/2
            Wt = W/2
            # move right boundary point to second spot (needed for DoF ordering in element below)
            Xt[:,[1,-1]] = Xt[:,[-1,1]]
            Wt[[1,-1]] = Wt[[-1,1]]
            self.GL_quad[k] = ( Xt, Wt )

    def __call__(self, pp):
        return self.GL_quad[pp]

class ElementLinePp_nodal(ElementH1):
    """Piecewise nodal :math:`p`'th order element."""

    nodal_dofs = 1
    refdom: Type[Refdom] = RefLine

    def __init__(self, p):
        if p < 1:
            raise ValueError("p < 1 not supported.")
        if p < 3:
            logger.warning(("Replace ElementLinePp_nodal({}) by ElementLineP{}() "
                            "for performance.").format(p, p))

        self.interior_dofs = p - 1
        self.maxdeg = p
        self.dofnames = ['u'] + (p - 1) * ['u']
        self.quad = GLL_quad()(p + 1)
        self.doflocs = self.quad[0].T
        self.P = np.zeros((0, 0))
        self.dP = np.zeros((0, 0, 1))
        self.p = p

        # get a barycentric interpolator for each Lagrange basis function y_ik = delta_ik
        self.GLLI = BarycentricInterpolator(xi=self.quad[0][0], yi=np.eye(p + 1))

    def _reval_legendre(self, y, p):
        """Re-evaluate Legendre polynomials."""
        P = np.zeros((p + 1,) + y.shape)
        dP = np.zeros((p + 1, 1) + y.shape)

        P = np.moveaxis(self.GLLI(y), -1, 0)
        dP = np.moveaxis(self.GLLI.derivative(y)[None,...], -1, 0)

        return P, dP

    def lbasis(self, X, i):

        if self.P.shape[1] != X.shape[1]:
            self.P, self.dP = self._reval_legendre(X[0, :], self.p)

        return self.P[i], self.dP[i]
