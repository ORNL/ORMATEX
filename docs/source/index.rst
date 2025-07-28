.. ormatex documentation master file, created by
   sphinx-quickstart on Mon May 17 16:22:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ORMATEX's documentation!
===================================

Oak Ridge Matrix Exponential toolkit.

ORMATEX contains methods to compute the matrix exponential:  :math:`\mathrm{exp}(A t)`, and the action of the matrix exponential on a vector: :math:`\mathrm{exp}(A t)v_0`, where :math:`A` is a matrix.  Additionally, this package contains related methods for the :math:`\varphi`-functions.  Krylov methods to evaluate the matrix exponential-vector and :math:`\varphi`-vector products are provided for cases where :math:`A` is large and sparse.
Utilizing these methods, ORMATEX implements performant exponential integrators for large systems of coupled ODEs.

ORMATEX is a mixed Rust and Python package that provides an extensible foundation to construct advanced exponential integrators.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   intro
   quick_start
   examples
   ormatex_py
