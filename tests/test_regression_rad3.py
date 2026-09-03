"""
Regression test for 3 species Reaction-Advection-Diffusion system

Check that all exponential integrators solve a simple
RAD system to the expected precission.
"""
from ormatex_py import integrate_wrapper
from ormatex_py.progression.rad_1d_3s import RAD_SEM
from ormatex_py.progression.advection_diffusion_1d import AdDiffSEM, torus_distance
from ormatex_py.progression.bateman_sys import gen_bateman_matrix, gen_transmute_matrix, analytic_bateman_single_parent
from ormatex_py.ode_utils import stack_u, flatten_u
import skfem as fem
import numpy as np
import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

try:
    from ormatex_py.ormatex import complex_diag_leja_phikv_static_rs
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False


def _rad_3s(method="exprb2", phi_method="krylov"):
    # create the mesh
    dwidth = 1.0
    mesh0 = fem.MeshLine1(np.array([[0., dwidth]])).with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], dwidth)
    })
    # mesh refinement
    mr = 7
    mesh = mesh0.refined(mr)
    mesh = fem.MeshLine1DG.periodic(
        mesh,
        mesh.boundaries['right'],
        mesh.boundaries['left'],
    )
    # order
    p = 2

    # velocity and diffusion coefficient
    vel, nu = 0.5, 1.0e-5
    param_dict = {"nu": nu, "vel": vel}

    n_species = 3
    sem = AdDiffSEM(mesh, p=p, params=param_dict)
    ode_sys = RAD_SEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    # initial profiles for each species
    gauss_scale = 1.0
    wc, ww = 0.5, 0.05
    var = ww ** 2.0
    g_prof0 = lambda x: 0.0*x + 1e-16
    def g_prof_exact(t, x):
        out = np.zeros(x.shape)
        shifts = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) * dwidth
        ns = len(shifts)
        for s in shifts:
            out += np.exp(-(s + torus_distance(x-t*vel, wc))**2.0 / (4*var+4*nu*t))
        norm_const = np.sqrt(4*var) / (np.sqrt((4*var+4*nu*t)))
        out *= norm_const
        out *= gauss_scale
        return out
    g_prof1 = lambda x: g_prof_exact(0.0, x)
    y0_profile = [
            g_prof1(xs),
            g_prof0(xs),
            g_prof0(xs),
    ]
    y0 = flatten_u(jnp.asarray(y0_profile).transpose())

    # time step settings
    t0 = 0.0
    dt = 0.1
    tf = 0.3
    nsteps = int(np.round(tf / dt))

    # setup the species reactions
    keymap = ["c_0", "c_1", "c_2"]
    decay_lib = {
        'c_0':  ('c_1', 1.0e-1),
        'c_1':  ('c_2', 1.0e1),
        'c_2':  ('none', 1.0e-2),
    }
    bat_mat = gen_bateman_matrix(keymap, decay_lib)

    # Compute analytic solution.
    ts = np.linspace(0.0, nsteps*dt, nsteps+1)
    scale_true = analytic_bateman_single_parent(ts, bat_mat, 1.0)
    profile_true = []
    for i, t in enumerate(ts):
        prof = scale_true[i].reshape((-1,1)) @ g_prof_exact(t, ode_sys.xs).reshape((-1,1)).T
        profile_true.append(prof)
    y_true = np.asarray(profile_true)

    # integrate the system
    res = integrate_wrapper.integrate(
            ode_sys, y0, t0, dt, nsteps, method, phi_method=phi_method,
            max_krylov_dim=120, iom=2, tol=1e-12)
    t_res, y_res = res.t_res, res.y_res

    # check the final result
    u_true = y_true[-1].T
    u_res = stack_u(y_res[-1], len(keymap))
    tol = 1e-2
    abs_diff = np.abs(u_res[:, 2] - u_true[:, 2])
    assert np.mean(abs_diff / np.mean(u_true[:, 2])) < tol


def test_rad_3s():
    _rad_3s("exprb3", "krylov")
    _rad_3s("epi3", "krylov")
    _rad_3s("exprb3", "leja")
    _rad_3s("epi3", "leja")
    _rad_3s("exprb3", "pfd")
    _rad_3s("exprb3_pfd", "pfd")
    _rad_3s("exp3_dense", "pfd")
    if HAS_ORMATEX_RUST:
        _rad_3s("epi3_rs", phi_method="krylov")
        _rad_3s("epi3_rs", phi_method="leja")
