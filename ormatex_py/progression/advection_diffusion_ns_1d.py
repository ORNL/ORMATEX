"""
Multiple species 1D advection diffusion case.

- Supports N numbers of coupled species.
- Each species can have unique, space, time, and concentration
dependant non-linear sources and sinks.

Similar to the single species case (advection_diffusion_1d.py);
however, internally the species vector, `u` is now a 2D tensor,
with each column representing a new species.

All species are assumed to be advected according to the same
velocity field.  All species share the same bulk diffusion coefficient.

We can write:
    U_t = L*u + v*\nabla U - D \nabla^2 U + S(U)

where

    U = {u_0, u_1, ..., u_N}

    or in matrix form

U = [
     [u_{0,0}, ... u_{N,0},],
     [u_{0,1},     u_{N,1} ],
     [...      ...         ],
     [u_{0,X}  ... u_{N,X} ],
]
"""
import numpy as np
import scipy as sp
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
import equinox as eqx

import skfem as fem
from skfem.helpers import dot, grad

from ormatex_py.ode_sys import OdeSys, FdJacLinOp

from ormatex_py.progression.bateman_sys import gen_bateman_matrix
from ormatex_py.progression import element_line_pp_nodal as el_nodal
from ormatex_py.progression.advection_diffusion_1d import mass
from ormatex_py.progression import integrate_wrapper

# Specify velocity
vel = 0.5

# decay lib
decay_lib = {
    'u_0':  ('u_1', 3.0e-1),
    'u_1':  ('none', 0.3e-1),
}

def src_f(x, **kwargs):
    """
    Custom source term, could depend on solution y
    """
    return 0.0 * np.exp(- np.sum((x - 0.2) / 0.1**2, axis=0)**2 / 2)


def vel_f(x, **kwargs):
    """
    Custom velocity field
    """
    return 0.0*x + vel

@eqx.filter_jit
def stack_u(u: jax.Array, n: int):
    return u.reshape((-1, n), order='F')

@eqx.filter_jit
def flatten_u(u: jax.Array):
    # use column major ordering
    return u.flatten(order='F')

@fem.BilinearForm
def adv_diff(u, v, w):
    """
    Combo Adv Diff kernel

    Args:
        w: is a dict of skfem.element.DiscreteField
            w.x (dof locs)
    """
    return w.nu * dot(grad(u), grad(v)) + dot(vel_f(w.x), grad(u)) * v

@fem.BilinearForm
def adv_diff_cons(u, v, w):
    """
    Combo Adv Diff kernel conservative form
    """
    return w.nu * dot(grad(u), grad(v)) - dot(vel_f(w.x) * u, grad(v))

@fem.LinearForm
def rhs(v, w):
    """
    Source term
    """
    return src_f(w.x, w=w) * v

@fem.BilinearForm
def robin(u, v, w):
    """
    Args:
        w: is a dict of skfem.element.DiscreteField (or user types)
            w.n (face normals)
    """
    return dot(vel_f(**w), w.n) * u * v


class AdDiffSEM:
    """
    Assemble matrices for spectral finite element discretization
    of an advection diffusion eqaution

    p: polynomial order
    nu: diffusion coeff
    """
    def __init__(self, mesh, p=1, n_species=1, field_fns={}, params={}, **kwargs):
        # diffusion coeff
        self.params = {"nu": params.get("nu", 5e-3)}
        self.p = p
        self.mesh = mesh
        self.n_species = n_species

        # register custom field functions
        self.field_fns = {}
        for k, v in field_fns.items():
            self.validate_add_field_fn(k, v)

        if self.p < 2:
            self.element = fem.ElementLineP1()
        elif self.p == 2:
            self.element = fem.ElementLineP2()
        elif self.p >= 3:
            self.element = el_nodal.ElementLinePp_nodal(self.p)

        quad = el_nodal.GLL_quad()(self.p + 1)
        self.basis = fem.Basis(self.mesh, self.element, quadrature=quad)
        if self.mesh.boundaries:
            # if the mesh has boundaries, get a basis for BC
            self.basis_f = fem.FacetBasis(self.mesh, self.element)

    def validate_add_field_fn(self, field_name: str, f: Callable):
        assert callable(f)
        self.field_fns[field_name] = f

    def w_ext(self, basis, reshape=True, **kwargs):
        """
        Extra kwargs passed with the `w` dict to kernels
        """
        species_id = {"species_id": kwargs.get("species_id", -1),
                      "n_species": self.n_species}
        fields = {}
        basis_shape = basis.global_coordinates().shape
        # for nonlinear source and sink terms
        u = kwargs.get("u", None)
        if u is not None and species_id["species_id"] >= 0:
            for n in range(self.n_species):
                ub = basis.interpolate(u.at[:, n].get())
                fields["u_%d"%n] = ub
        # other aux fields
        for field_name, field_f in self.field_fns.items():
            vals = field_f(basis.doflocs, **kwargs).flatten()
            # NOTE: reshape is needed here
            # when using these fields in the kernels.
            # is this an issue in skfem??
            fv = basis.interpolate(vals)
            if reshape:
                fv = np.reshape(fv, basis_shape)
            fields[field_name] = fv
        full_w_dict = {**self.params, **fields, **species_id}
        return full_w_dict

    def assemble(self, **kwargs):
        """
        kwargs: extra args passed to user defined field functions
        """
        conservative = True
        if conservative:
            # remark: w_dict must contain only scalars and skfem.element.DiscreteField
            self.A = adv_diff_cons.assemble(self.basis, **self.w_ext(self.basis, **kwargs))
            if self.mesh.boundaries:
                # if not periodic, add boundary term
                self.A += robin.assemble(self.basis_f, **self.w_ext(self.basis_f, **kwargs))
        else:
            self.A = adv_diff.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        M = mass.assemble(self.basis, **self.w_ext(self.basis, **kwargs))

        if self.mesh.boundaries:
            # Dirichlet boundary conditions
            fem.enforce(M, D=self.mesh.boundaries['left'].flatten(), overwrite=True)
            fem.enforce(self.A, D=self.mesh.boundaries['left'].flatten(), overwrite=True)
        else:
            # periodic boundary conditions
            fem.enforce(M, D=self.basis.get_dofs().flatten(), overwrite=True)
            fem.enforce(self.A, D=self.basis.get_dofs().flatten(), overwrite=True)

        Ml = M @ np.ones((M.shape[1],))

        # provide jax arrays
        jMl = jnp.asarray(Ml)
        jA = jsp.BCOO.from_scipy_sparse(self.A)
        return jA, jMl

    def assemble_rhs(self, u=None, **kwargs):
        """
        Handle nonlinear source term.  Call if the solution updates
        and the rhs depends on u.
        """
        assert hasattr(self, 'A')
        wkwargs = {**kwargs, **{'u': u}}
        b = []
        for n in range(self.n_species):
            # unique rhs for each species
            bn = rhs.assemble(self.basis, **self.w_ext(self.basis, species_id=n, **wkwargs))
            b.append(bn)
        if self.mesh.boundaries:
            # Dirichlet boundary conditions
            for bn in b:
                fem.enforce(self.A, bn, D=self.mesh.boundaries['left'].flatten(), overwrite=True)
        else:
            # periodic boundary conditions
            for bn in b:
                fem.enforce(self.A, bn, D=self.basis.get_dofs().flatten(), overwrite=True)
        # columns index in b represents each species id
        jb = jnp.asarray(b).transpose()
        return jb

    def xs(self):
        return self.basis.doflocs[0]

    def ode_sys(self, **kwargs):
        return AffineLinearSEM(self, **kwargs)


class AffineLinearSEM(OdeSys):
    """
    Define ODE System associated to affine linear sparse Jacobian problem

    due to the non-autodifferentiable, non-jitable assemble call
    use jax_disable_jit to use this class
    """
    bat_mat: jax.Array
    A: jsp.JAXSparse
    Ml: jax.Array
    sys_assembler: AdDiffSEM

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        self.sys_assembler = sys_assembler
        self.A, self.Ml = sys_assembler.assemble(**kwargs)
        self.bat_mat = gen_bateman_matrix(['u_0', 'u_1'], decay_lib)
        super().__init__()

    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        n = self.bat_mat.shape[0] #self.sys_assembler.n_species
        un = stack_u(u, n)
        # nonlin rhs
        b = self.sys_assembler.assemble_rhs(u=un)
        # add bateman
        lub = un @ self.bat_mat.transpose()
        udot = (b - self.A @ un) / self.Ml.reshape((-1, 1)) + lub
        # integrators currently expect a flat U
        return flatten_u(udot)

    def _fjac(self, t: float, u: jax.Array, **kwargs):
        # finite difference Jacobian
        # automatic Jacobian is not supported
        return FdJacLinOp(t, u, self.frhs, frhs_kwargs=kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    jax.config.update("jax_disable_jit", True)
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-ic", help="one of [square, gauss]", type=str, default="gauss")
    parser.add_argument("-mr", help="mesh refinement", type=int, default=6)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-per", help="impose periodic BC", action='store_true')
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    args = parser.parse_args()

    # create the mesh
    mesh0 = fem.MeshLine1().with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], 1.)
    })
    # mesh refinement
    nrefs = args.mr
    mesh = mesh0.refined(nrefs)

    periodic = args.per
    if periodic:
        mesh = fem.MeshLine1DG.periodic(
            mesh,
            mesh.boundaries['right'],
            mesh.boundaries['left'],
        )

    # diffusion coefficient
    param_dict = {"nu": 1e-8, "vel": vel}

    # init the system
    n_species = 2
    sem = AdDiffSEM(mesh, p=args.p, n_species=n_species, params=param_dict)
    ode_sys = AffineLinearSEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    # initial profiles for each species
    wc, ww = 0.3, 0.05
    g_prof = lambda x: np.exp(-((x-wc)/(2*ww))**2.0)
    g_prof2 = lambda x: 0.2*np.exp(-((x-wc)/(2*ww))**2.0)
    y0_profile = [g_prof(xs), g_prof2(xs)]
    y0 = flatten_u(jnp.asarray(y0_profile).transpose())

    # integrate the system
    t0 = 0.
    dt = .1
    nsteps = 20
    method = args.method
    t_res, y_res = integrate_wrapper.integrate(ode_sys, y0, t0, dt, nsteps, method, max_krylov_dim=100)

    si = xs.argsort()
    sx = xs[si]
    plt.figure()
    for i in range(nsteps):
        if i % 2 == 0 or i == 0:
            t = t_res[i]
            yf = y_res[i]
            uf = stack_u(yf, n_species)
            for n in range(n_species):
                us = uf[:, n]
                line_style = None
                if (n+1) % 2 == 0:
                    line_style = '--'
                plt.plot(sx, us[si], ls=line_style, label='t=%0.4f, species=%s' % (t, str(n)))
    plt.legend()
    plt.grid(ls='--')
    plt.ylabel('u')
    plt.xlabel('x')
    plt.title(r"method: %s" % (method))
    plt.savefig('adv_diff_ns_1d.png')
    plt.close()
