Introduction
============

The current set of implemented and planned time integration methods in
each language:

Rust
~~~~

Exponential integrators:

-  ☒ EPI2
-  ☒ EPI3
-  ☐ EPIRK4
-  ☒ EXPRB2
-  ☒ EXPRB3
-  ☐ PEXPRB4 (parallel exponential rosenbrock order 4)

Classic integrators:

-  ☒ RK1,RK2,RK3,RK4
-  ☒ Backward Euler
-  ☒ BDF2
-  ☒ Crank-Nicolson
-  ☐ DIRK
-  ☐ SDIRK

Python
~~~~~~

Exponential integrators:

Jacobian based:

-  ☒ EPI2
-  ☒ EPI3
-  ☐ EPIRK4
-  ☒ EXPRB2
-  ☒ EXPRB3
-  ☒ PEXPRB4

Splitting linear operator based:

-  ☒ EXP1
-  ☒ EXP2
-  ☒ EXP3
-  ☐ EXP4

Classic integrators:

-  ☒ all explicit and implicit integrators supported by diffrax (through
   interface with diffrax)

Methods Reference
-----------------

The following integrators are available through the common high level
``integrate_wrapper.integrate`` interface. Different integrators can be
specified through the ``method`` keyword argument.

.. csv-table::
    :header: method , order , Impl Notes , kwargs , description , Reference

    ``exprb2``, 2 , JAX/python , max_krylov_dim; iom , Exponential Rosenbrock order 2, https://doi.org/10.1137/080717717
    ``exprb3``, 3 , JAX/python , max\_krylov\_dim; iom , Exponential Rosenbrock order 3, https://doi.org/10.1137/080717717 
    ``pexprb4``, 4 , JAX/python , max\_krylov\_dim; iom , Parallel Exponential Rosenbrock order 4, https://doi.org/10.1016/j.camwa.2016.01.020 
    ``epi3``, 3 , JAX/python , max\_krylov\_dim; iom , Exponential Propagation Iterative order 3, https://doi.org/10.1137/110849961 
    ``rk4`` , 4 , JAX/python , , Explicit RK4  , 
    ``implicit_euler`` , 1 , JAX/diffrax ,  , Backward Euler , 
    ``implicit_esdirk3``, 3 , JAX/diffrax , , explicit singly diagonal implicit order 3 , 
    ``dopri5`` , 5 , JAX/diffrax , , Explicit Dormand-Prince order 5  , 
    ``exprb2_rs``, 2 , Rust , max\_krylov\_dim; iom , Exponential Rosenbrock order 2, https://doi.org/10.1137/080717717 
    ``exprb3_rs``, 3 , Rust , max\_krylov\_dim; iom , Exponential Rosenbrock order 3, https://doi.org/10.1137/080717717 
    ``epi3_rs``, 3 , Rust , max\_krylov\_dim; iom , Exponential Propagation Iterative order 3, https://doi.org/10.1137/110849961 
    ``bdf1_rs``, 1 , Rust ,  , Backward Euler , 
    ``bdf2_rs``, 2 , Rust ,  , Backward difference formula 2, 
    ``cn_rs``, 2 , Rust ,  , Crank-Nicolson , 
    ``rk1_rs``, 1 , Rust ,  , Forward Euler , 
    ``rk4_rs``, 4 , Rust ,  , Explicit RK4 , 

The Rust-based integrators can be accessed through the common python ``integrate_wrapper.integrate`` interface after the Rust-Python bindings are built and installed.  Alternatively, the Rust integrator implementations can be used directly from a Rust-based program.
