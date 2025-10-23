"""
This example solves a system of three advected, coupled reactive species.  This example only contains linear Bateman decay
for which an analytic solution may be developed.

The purpose of this example is to benchmark the various
exponential integration methods to ensure the expected accuracy
and order of accuracy is obtained when compared with
the analytic result.

NOTE: When running on a CPU target, it is recommended to
use the following env variables for best performance:

    OMP_NUM_THREADS=4 XLA_FLAGS=--xla_cpu_use_thunk_runtime=false python rad_1d_3s.py <args>

Ref: https://github.com/jax-ml/jax/discussions/25711
"""
import numpy as np
import scipy as sp
from itertools import cycle
import os

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
from jax.experimental.sparse import BCSR
import equinox as eqx

import skfem as fem

from ormatex_py.ode_sys import OdeSys, OdeSplitSys, MatrixLinOp
from ormatex_py.ode_utils import stack_u, flatten_u
from ormatex_py.matexp_leja import plot_leja_conjugate_ellipse_error

from ormatex_py.progression.species_source_sink import mxf_liq_vapor_bubble_ig, mxf_arrhenius, mxf_liq_vapor_nonlin
from ormatex_py.progression.advection_diffusion_1d import AdDiffSEM
from ormatex_py.progression.bateman_sys import gen_bateman_matrix, gen_transmute_matrix, analytic_bateman_single_parent
from ormatex_py import integrate_wrapper

keymap = ["c_0", "c_1", "c_2"]
decay_lib = {
    'c_0':  ('c_1', 1.0e-1*10),
    'c_1':  ('c_2', 1.0e1*10),
    'c_2':  ('none', 1.0e-3*10),
}

outdir = "./rad_1d_3s_out/"
if not os.path.exists(outdir): os.mkdir(outdir)

class RAD_SEM(OdeSplitSys):
    """
    Define ODE System associated to RAD problem
    """
    bat_mat: jax.Array
    A: jsp.JAXSparse
    Ml: jax.Array
    xs: jax.Array

    dirichlet_bd: np.array

    def __init__(self, sys_assembler: AdDiffSEM, *args, **kwargs):
        # get stiffness matrix and mass vector
        A, self.Ml, _ = sys_assembler.assemble(**kwargs)
        self.A = BCSR.from_bcoo(A)
        # get collocation points
        self.xs = sys_assembler.collocation_points()

        # Dirichlet boundary conditions
        if sys_assembler.dirichlet_bd is not None:
            self.dirichlet_bd = sys_assembler.dirichlet_bd
        else:
            self.dirichlet_bd = np.array([], dtype=int)

        # get bateman matrix
        self.bat_mat = gen_bateman_matrix(keymap, decay_lib)
        super().__init__()

    @jax.jit
    def _frhs(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        n = self.bat_mat.shape[0]
        un = stack_u(u, n)
        # bateman
        lub = un @ self.bat_mat.transpose()
        # full system
        udot = lub - (self.A @ un) / self.Ml.reshape((-1, 1))
        # set the time derivative of the Dirichlet boundary data to zero
        udot = udot.at[self.dirichlet_bd,:].set(0.)
        # integrators currently expect a flat U
        return flatten_u(udot)

    @jax.jit
    def _fl(self, t: float, u: jax.Array, **kwargs) -> jax.Array:
        """
        define a linear operator (for testing purposes)
        # TODO: uses dense Kronecker product, should be a BlockDiagLinOp (not implemented yet)
        """
        n = self.bat_mat.shape[0]
        Ndof = self.Ml.shape[0]
        # only use Bateman terms for L
        #L = jnp.kron(self.bat_mat, jnp.eye(Ndof))
        # only use adv-diff terms for L
        L = jnp.kron(jnp.eye(n), (- self.A / self.Ml.reshape((-1, 1))).todense())
        #print(L)
        return MatrixLinOp(L)

def plot_dt_jac_spec(ode_sys, y, t=0.0, dt=1.0, figname="reac_adv_diff_s3_eigplot"):
    """
    Plots eigvals of the scaled system Jacobian
    """
    import matplotlib.pyplot as plt
    dtJ = np.asarray(dt*ode_sys.fjac(t, y).dense())
    print("dt*J", dtJ)
    eigdtJ = np.linalg.eig(dtJ)[0]

    plt.figure()
    plt.scatter(-eigdtJ.real+1., eigdtJ.imag)
    plt.ylabel('Imaginary')
    plt.xlabel('(-)Real + 1')
    plt.xscale('log')
    plt.grid(alpha=0.5, ls='--')
    plt.title(r"$\Delta$t*Jac eigenvalues. $\Delta$t=%0.3e" % dt)
    plt.tight_layout()
    plt.savefig(figname + ".png")
    plt.close()
    dtJnorm = np.linalg.norm(dtJ, ord=np.inf)
    dtJeig_max = np.max(np.abs(eigdtJ))
    dtJeig_min = np.min(np.abs(eigdtJ))
    print("CFL: %0.4f/%0.4f" % (dtJeig_max, dtJnorm))
    return dtJeig_max, dtJeig_min

def plot_leja_conv_detail(
        ode_sys, y, t, dt, outdir="./",
        n_leja_list=[4, 8, 12, 24, 36, 50, 76, 100, 126, 150, 176, 200, 240],
        **kwargs):
    """
    Plots leja polynomial convergence details
    """
    import matplotlib.pyplot as plt
    dtJ = np.asarray(dt*ode_sys.fjac(t, y).dense())
    eigdtJ = np.linalg.eig(dtJ)[0]
    a = kwargs.get("leja_a", None)
    a = np.min(eigdtJ.real) if a is None else a
    b = 0.0
    c = kwargs.get("leja_c", np.max(np.abs(eigdtJ.imag)))
    # differnet leja polynomial parameters
    leja_plist = {r"$\mathrm{Leja}_{CLaPM}\ l_1\ dd_{pade}$": {"c": c, "leja_n_zeros": 1, "dd_method": "pade"},
                  r"$\mathrm{Leja}_{CLaPM}\ l_1\ dd_{ts}$": {"c": c, "leja_n_zeros": 1, "dd_method": "taylor"},
                  r"$\mathrm{Leja}_{CLaPM}\ l_0\ dd_{pade}$": {"c": c, "leja_n_zeros": 0, "dd_method": "pade"},
                  r"$\mathrm{Leja}_{CLaPM}\ l_0\ dd_{ts}$": {"c": c, "leja_n_zeros": 0, "dd_method": "taylor"},
                  r"$\mathrm{Leja}_{ReLPM}\ l_0\ dd_{pade}$": {"c": 0.0, "leja_n_zeros": 0, "dd_method": "pade"},
                  r"$\mathrm{Leja}_{ReLPM}\ l_0\ dd_{ts}$": {"c": 0.0, "leja_n_zeros": 0, "dd_method": "taylor"},
                  r"$\mathrm{Leja}_{ReLPM}\ l_0\ dd_{rc}$": {"c": 0.0, "leja_n_zeros": 0, "dd_method": "recursive"},
                  }
    err_dict = {}
    for key, leja_p in leja_plist.items():
        l1_err_list, l2_err_list = [], []
        for n_leja in n_leja_list:
            i, l1_expmv_err, l2_expmv_err = plot_leja_conjugate_ellipse_error(
                    a=a, b=b, c=leja_p["c"], eigJ=eigdtJ, leja_n_zeros=leja_p["leja_n_zeros"],
                    v=y, dd_method=leja_p['dd_method'],
                    n_leja=n_leja, leja_tol=1e-30, dirname=outdir)
            l1_err_list.append((i, l1_expmv_err))
            l2_err_list.append((i, l2_expmv_err))
            if l1_expmv_err < 1e-12:
                break
        err_dict[key] = (np.asarray(l1_err_list), np.asarray(l2_err_list))
    # plot expm err as fn of number of leja points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ls_cycler = cycle(['-', '--'])
    for key, (l1_err_list, l2_err_list) in err_dict.items():
        ls = next(ls_cycler)
        ax1.plot(l1_err_list[:, 0], l1_err_list[:, 1], alpha=1.0, ls=ls, label=key)
        ax2.plot(l2_err_list[:, 0], l2_err_list[:, 1], alpha=1.0, ls=ls, label=key)
    ax1.grid(ls='--')
    ax2.grid(ls='--')
    ax1.set_ylabel(r"$|e^{A} v - p_{leja}|_\infty$ err")
    ax2.set_ylabel(r"$||e^{A} v - p_{leja}||_2$ err")
    ax1.set_xlabel("N leja points")
    ax2.set_xlabel("N leja points")
    plt.tight_layout()
    plt.legend()
    plt.savefig(outdir + "/leja_converge.png", dpi=200)
    plt.close()

def main(dt, method='epi3', periodic=True, mr=6, p=2, tf=1.0, jac_plot=False, nu=0.002, **kwargs):
    # create the mesh
    dwidth = 1.0
    mesh0 = fem.MeshLine1(np.array([[0., dwidth]])).with_boundaries({
        'left': lambda x: np.isclose(x[0], 0.),
        'right': lambda x: np.isclose(x[0], dwidth)
    })
    # mesh refinement
    nrefs = mr
    mesh = mesh0.refined(nrefs)

    if periodic:
        mesh = fem.MeshLine1DG.periodic(
            mesh,
            mesh.boundaries['right'],
            mesh.boundaries['left'],
        )

    # diffusion coefficient
    vel = 0.5
    param_dict = {"nu": nu, "vel": vel}

    # init the system
    n_species = 3
    sem = AdDiffSEM(mesh, p=p, params=param_dict)
    ode_sys = RAD_SEM(sem)
    t = 0.0

    # mesh mask for initial conditions
    xs = np.asarray(sem.basis.doflocs.flatten())

    # initial profiles for each species
    wc, ww = 0.4, 0.05
    var = ww ** 2.0
    g_prof0 = lambda x: 0.0*x + 1e-16
    fix_scale = True
    gauss_scale = 1.0
    if fix_scale:
        gauss_scale = (1./ np.sqrt(1.*(var/2))) ** -1.0
    if periodic:
        g_prof1 = lambda x: \
            (
            np.exp(-((1.0-((x - wc) % 1))**2.0 / (2*var))) + \
            np.exp(-((((x - wc) % 1)**2.0) / (2*var)))
            ) * (1./ np.sqrt(1.*(var/2))) * gauss_scale
        g_prof_exact = lambda t, x: \
            (
            np.exp(-((1.0-((x - (wc+t*vel)) % 1))**2.0 / (2*var+4*nu*t)) ) + \
            np.exp(-((((x - (wc+t*vel)) % 1))**2.0 / (2*var+4*nu*t)) )
            ) * (1./np.sqrt(1.*(var/2+nu*t))) * gauss_scale
    else:
        g_prof1 = lambda x: np.exp(-((x-wc)**2/(2*var))) * (1./ np.sqrt(1.*(var/2))) * gauss_scale
        g_prof_exact = lambda t, x: np.exp(-((x-(wc+t*vel))**2.0 / (2*var+4*nu*t))) \
                * (1./np.sqrt(1.*(var/2+nu*t))) * gauss_scale
    y0_profile = [
            g_prof1(xs),
            g_prof0(xs),
            g_prof0(xs),
    ]
    y0 = flatten_u(jnp.asarray(y0_profile).transpose())

    # time step settings
    t0 = 0.0
    dt = dt
    nsteps = int(np.round(tf / dt))

    # Compute analytic solution. In this case,
    # the analytic solution is the product of pure bateman decay
    # solution with the advected gaussian wave.
    bat_mat = gen_bateman_matrix(keymap, decay_lib)
    print("bateman mat:")
    print(bat_mat)
    ts = np.linspace(0.0, nsteps*dt, nsteps+1)
    scale_true = analytic_bateman_single_parent(ts, bat_mat, 1.0)
    profile_true = []
    for i, t in enumerate(ts):
        prof = scale_true[i].reshape((-1,1)) @ g_prof_exact(t, ode_sys.xs).reshape((-1,1)).T
        profile_true.append(prof)
    y_true = np.asarray(profile_true)

    # integrate the system
    res = integrate_wrapper.integrate(
            ode_sys, y0, t0, dt, nsteps, method, max_krylov_dim=200, iom=10, **kwargs)
    t_res, y_res = res.t_res, res.y_res

    si = xs.argsort()
    sx = xs[si]

    pfd_method = kwargs.get("pfd_method", '')
    method_str = method if not pfd_method else "%s %s" % (method, pfd_method)

    def plot_sol(plot_idx=-1):
        # Plot the solution at step plot_idx
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,10))
        mae_list, mae_rl_list = [], []

        t = t_res[plot_idx]
        yf = y_res[plot_idx]
        uf = stack_u(yf, n_species)
        ut = y_true[plot_idx]
        plot_data = []
        plot_data.append(sx)
        for n in range(0, n_species):
            us = uf[:, n]
            u_true = ut[n, :]
            u_calc = us[si]
            u_analytic = u_true[si]
            plot_data.append(u_analytic)
            plot_data.append(u_calc)
            ax[n].plot(sx, u_calc, label='t=%0.4f, species=%s' % (t, str(n)))
            ax[n].plot(sx, u_analytic, ls='--', label='t=%0.4f, true' % (t))
            # ax[n].set_yscale('log')
            ax[n].legend()
            ax[n].grid(ls='--')
            # compute diff
            diff = u_calc - u_analytic
            diff_rl = (u_calc - u_analytic) / (np.max(u_analytic))
            mae = np.mean(np.abs(diff))
            mae_rl = np.mean(np.abs(diff_rl))
            mae_list.append(mae)
            mae_rl_list.append(mae_rl)
            ax[n].set_title(r"%s, MAE: %0.3e, $\Delta$t=%0.2e" % (method_str, mae, dt))
        ax[0].set_ylabel("Species 0 [mol/cc]")
        ax[1].set_ylabel("Species 1 [mol/cc]")
        ax[2].set_ylabel("Species 2 [mol/cc]")
        ax[0].set_xlabel("location [m]")
        plt.tight_layout()
        plt.savefig(outdir + 'reac_adv_diff_s3_%s_%0.2f_%1.3f.png' % (method_str, t, dt))
        plt.close()
        header = "x,u0_analytic,u0_calc,u1_analytic,u2_calc,u2_analytic,u2_calc"
        np.savetxt(outdir + 'reac_adv_diff_s3_%s_%0.2f_%1.3f.txt' % (method_str, t, dt), np.asarray(plot_data).T, delimiter=', ', header=header)
        return mae_list, mae_rl_list

    _, _ = plot_sol(0)
    mae_list, mae_rl_list = plot_sol(-1)

    if jac_plot:
        # plot the jacobian spectrum
        plot_dt_jac_spec(ode_sys, y_res[-1], 0.0, dt, figname="reac_adv_diff_s3_eigplot")

        # plot the leja polynomial matexp-vec approx
        if "leja" in method and kwargs.get("leja_plot", False):
            plot_leja_conv_detail(ode_sys, y_res[-1], t, dt, outdir=outdir, **kwargs)

    print("=== Species MAEs at t=%0.4e ===" % t_res[-1])
    [print("%0.4e" % a, end=', ') for a in mae_list]
    print()
    si = xs.argsort()
    sx = xs[si]
    mesh_spacing = (sx[1] - sx[0])
    cfl = dt * vel / mesh_spacing
    print("mesh_spacing: %0.4e, CFL=%0.4f" % (mesh_spacing, cfl))
    return mae_list, mae_rl_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    jax.config.update("jax_enable_x64", True)
    print(f"Running on {jax.devices()}.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-sweep", help="run method sweep", default=False, action='store_true')
    parser.add_argument("-mr", help="mesh refinement", type=int, default=6)
    parser.add_argument("-p", help="basis order", type=int, default=2)
    parser.add_argument("-dt", help="time step size", type=float, default=0.1)
    parser.add_argument("-leja_tol", help="optional leja integrator tolerance", type=float, default=1.0e-15)
    parser.add_argument("-leja_a", help="optional min real part of the J*dt spectrum. If None, power iter is used to determine this value.", type=float, default=None)
    parser.add_argument("-leja_c", help="optional max complex part of the J*dt spectrum", type=float, default=1.0)
    parser.add_argument("-leja_substep", help="optional to enable substepping the leja integrator", action='store_true', default=False)
    parser.add_argument("-leja_plot", help="additional leja polynomial convergence plots", default=False, action='store_true')
    parser.add_argument("-nu", help="diffusion coeff", type=float, default=1e-10)
    parser.add_argument("-tf", help="final time", type=float, default=1.0)
    parser.add_argument("-per", help="impose periodic BC", action='store_true')
    parser.add_argument("-method", help="time step method", type=str, default="epi3")
    parser.add_argument("-dd_method", help="divided difference method", type=str, default="taylor")
    parser.add_argument("-leja_n_zeros", help="number of zeros prepended to leja sequence", type=int, default=1)
    parser.add_argument("-nojit", help="Disable jax jit", default=False, action='store_true')
    args = parser.parse_args()
    jax.config.update("jax_disable_jit", args.nojit)

    if args.sweep:
        methods = ['exprb2', 'epi2_leja_im', 'exprb2_pfd', 'exprb2_pfd',
                   'exprb2_pfd', 'exprb2_pfd', 'implicit_esdirk3']
        pfd_methods = ['', '', 'pade_1_2', 'pade_2_4', 'cram_6', 'cram_16', '']
        dts = [0.025, 0.05, 0.1, 0.125, 0.2, 0.5]
        mae_sweep = {}
        mae_rl_sweep = {}
        for method, pfd_method in zip(methods, pfd_methods):
            method_str = method if not pfd_method else "%s %s" % (method, pfd_method)
            mae_sweep[method_str] = []
            mae_rl_sweep[method_str] = []
            for dt in dts:
                mae, mae_rl = main(dt, method, True, args.mr, args.p,
                                   tf=args.tf, pfd_method=pfd_method, nu=args.nu,
                                   leja_a=args.leja_a, leja_c=args.leja_c,
                                   leja_substep=args.leja_substep, leja_tol=args.leja_tol,
                                   leja_n_zeros=args.leja_n_zeros, leja_plot=args.leja_plot)
                mae_sweep[method_str].append(([dt] + mae))
                mae_rl_sweep[method_str].append(([dt] + mae_rl))
            print("=== Method: %s" % method_str)
            print("dt, MAE, ")
            for mae_res in mae_sweep[method_str]:
                print(*["%0.4e" % r for r in mae_res], sep=', ', end='')
                print()
            print("dt, MRE, ")
            for mae_rl_res in mae_rl_sweep[method_str]:
                print(*["%0.4e" % r for r in mae_rl_res], sep=', ', end='')
                print()

        # plot the MAE vs dt results for each method
        # color represents method, line style represents species number
        from matplotlib.pyplot import cm
        plt.figure()
        colors = iter(cm.rainbow(np.linspace(0, 1, len(methods))))
        for i, method_str in enumerate(mae_rl_sweep.keys()):
            dt_v_mae = np.asarray(mae_rl_sweep[method_str])
            color = next(colors)
            if i % 2 == 0:
                plt.plot(dt_v_mae[:, 0], dt_v_mae[:, 3], '-o', alpha=0.85, c=color, label="Method: %s" % method_str)
            else:
                plt.plot(dt_v_mae[:, 0], dt_v_mae[:, 3], ls='--', alpha=0.85, c=color, label="Method: %s" % method_str)
        plt.ylabel(r"Mean Relative Error. Avg((calc-true)/max(true))")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.xlabel(r"$\Delta$t")
        plt.grid(ls='--')
        plt.tight_layout()
        plt.savefig(outdir + "reac_adv_diff_s3_dt_err.png")
        plt.close()
    else:
        main(args.dt, args.method, args.per, args.mr, args.p,
             tf=args.tf, jac_plot=True, nu=args.nu,
             leja_a=args.leja_a, leja_c=args.leja_c, leja_substep=args.leja_substep,
             leja_tol=args.leja_tol, dd_method=args.dd_method, leja_n_zeros=args.leja_n_zeros,
             leja_plot=args.leja_plot)
