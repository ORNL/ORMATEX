import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    from ormatex_py.ormatex import complex_diag_leja_phikv_static_rs, complex_diag_leja_phikv_fitted_rs
    HAS_ORMATEX_RUST = True
except ImportError:
    HAS_ORMATEX_RUST = False


def plot_leja_conv_detail_rs(
        ode_sys, y, t, dt, outdir="./",
        n_leja_list=[4, 8, 12, 24, 36, 50], **kwargs):
    dtJ = np.asarray(dt*ode_sys.fjac(t, y).dense())
    eigdtJ = np.linalg.eig(dtJ)[0]

    a = kwargs.get("leja_a", None)
    a = np.min(eigdtJ.real) if a is None else a
    b = 0.0
    # c = kwargs.get("leja_c", np.max(np.abs(eigdtJ.imag)))
    c = np.max(np.abs(eigdtJ.imag))

    # create grid on the complex plane covering the spectrum
    # split into real and complex parts
    a_plot = (np.minimum(-1e-4, -np.max(np.abs(eigdtJ.real))) - 1) * 1.1
    b_plot = 0.0+1.0
    c_plot = (np.maximum(1e-4, np.max(np.abs(eigdtJ.imag))) + 1) * 1.1
    xr_grid = np.linspace(a_plot, b_plot, 200)
    xi_grid = np.linspace(-c_plot, c_plot, 200)
    zr_grid, zi_grid = np.meshgrid(xr_grid, xi_grid)
    zs_grid = zr_grid.flatten() + 1.j * zi_grid.flatten()
    d_diag = zs_grid
    d_diag_re = np.real(zs_grid)
    d_diag_im = np.imag(zs_grid)
    v = np.ones(len(d_diag_re), dtype=np.complex128) * np.linalg.norm(y)
    v_re = np.real(v)
    v_im = np.imag(v)

    x_diag_re = eigdtJ.real
    x_diag_im = eigdtJ.imag
    x = np.ones(len(x_diag_re), dtype=np.complex128) * np.linalg.norm(y)
    x_re = np.real(x)
    x_im = np.imag(x)

    # leja approx settings
    m = 180
    iom = 8
    krylov_reuse = kwargs.get("krylov_reuse", False)
    spec_iter = 24
    spec_saftey_factor = 1.05

    # compute expm(\Lambda)*v via leja approx
    y_col = np.atleast_2d(y).reshape(-1, 1)
    # TODO: Do not pass dtJ to fitted_rs, instead pass diag(eigs(dtJ))
    expmv_re, expmv_im, lp_sc_re, lp_sc_im = complex_diag_leja_phikv_fitted_rs(
            dtJ, np.ones(y_col.shape)*np.linalg.norm(y), 1.0,
            d_diag_re, d_diag_im, v_re, v_im,
            0, m, iom, spec_iter, krylov_reuse, spec_saftey_factor)
    # expmv_re, expmv_im, lp_sc_re, lp_sc_im = complex_diag_leja_phikv_static_rs(
    #         a, b, c, 1.0,
    #         d_diag_re, d_diag_im, v_re, v_im,
    #         0, m)
    expmv = np.vectorize(complex)(expmv_re, expmv_im)
    print("=== lp sequence detail ===")
    for i, (lp_re, lp_im) in enumerate(zip(lp_sc_re, lp_sc_im)):
        print(f"i: {i}, re: {lp_re}, im: {lp_im}")

    # compute the true result: expmv_true_i = exp(\lambda_i)*v_i
    # expmv_true = np.asarray(((np.exp(d_diag)-1.0) / d_diag) * v)
    expmv_true = np.asarray(np.exp(d_diag) * v_re)

    # compute the errors
    diff_grid = np.abs((expmv.flatten() - expmv_true.flatten()))

    # plot the errors on the complex plane
    Z = diff_grid.reshape(zr_grid.shape)

    # Compute errors at the eigenvalues
    expmv_x_re, expmv_x_im, _lp_sc_re, _lp_sc_im = complex_diag_leja_phikv_fitted_rs(
            dtJ, np.ones(y_col.shape)*np.linalg.norm(y), 1.0,
            x_diag_re, x_diag_im, x_re, x_im,
            0, m, iom, spec_iter, krylov_reuse, spec_saftey_factor)
    expmv_x = np.vectorize(complex)(expmv_x_re, expmv_x_im)
    expmv_x_true = np.asarray(np.exp(eigdtJ) * x_re)
    diff_x = np.abs((expmv_x.flatten() - expmv_x_true.flatten()))
    err_norm = np.max(np.abs(diff_x))

    xscale = "linear"
    plt.figure()
    if "log" in xscale:
        pcm = plt.pcolor(
                  -np.real(zs_grid.reshape(zr_grid.shape))+1,
                   np.imag(zs_grid.reshape(zr_grid.shape)), Z,
                   norm=colors.LogNorm(vmin=1e-14, vmax=100.0),
                   shading='auto')
        plt.scatter(-eigdtJ.real + 1., eigdtJ.imag, color='lightblue', marker='.',
                    label="Jacobian spectrum")
        plt.scatter(-np.real(lp_sc_re)+1, lp_sc_im,
                    color='tab:red', marker='x', label="Leja points")
        if krylov_reuse:
            plt.scatter(-np.real(lp_sc_re[0:spec_iter])+1, lp_sc_im[0:spec_iter],
                        color='tab:blue', marker='x', label=r"Ritz values")
        # plt.xscale(xscale)
        plt.xlabel("negative real + 1")
    else:
        pcm = plt.pcolor(
                   np.real(zs_grid.reshape(zr_grid.shape)),
                   np.imag(zs_grid.reshape(zr_grid.shape)), Z,
                   norm=colors.LogNorm(vmin=1e-14, vmax=100.0),
                   shading='auto')
        plt.scatter(eigdtJ.real, eigdtJ.imag, color='lightblue', marker='.',
                    label="Jacobian spectrum")
        plt.scatter(np.real(lp_sc_re), lp_sc_im,
                    color='tab:red', marker='x', label="Leja points")
        if krylov_reuse:
            plt.scatter(np.real(lp_sc_re[0:spec_iter]), lp_sc_im[0:spec_iter],
                        color='tab:blue', marker='x', label=r"Ritz values")
        plt.xlabel("real")
    plt.ylim(-c_plot, c_plot)
    plt.ylabel("imaginary")
    plt.title(r"m: %d $|P_{leja}(\Lambda)\mathbb{1} - exp({\Lambda}) \mathbb{1}|_\infty$: %0.4e" % (m, float(err_norm)))
    plt.colorbar(pcm)
    plt.legend(fancybox=True, framealpha=0.95, loc='lower right')
    plt.tight_layout()
    plt.savefig(outdir + "leja_rs_approx_err_contour_m_%d.png" % m, dpi=200)
    plt.close()


def plot_leja_conv_detail(
        ode_sys, y, t, dt, outdir="./",
        n_leja_list=[4, 8, 12, 24, 36, 50, 76, 100],
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
    leja_plist = {
                  r"$\mathrm{Leja}_{CLaPM}\ dd_{ts}$": {"a": a, "c": c, "leja_n_zeros": 0, "dd_method": "taylor"},
                  r"$\mathrm{Leja}_{ReLPM}\ dd_{ts}$": {"a": a, "c": 0., "leja_n_zeros": 0, "dd_method": "taylor"},
                  r"$\mathrm{Taylor}\ dd_{ts}$": {"a": 1e-8, "c": 0., "leja_n_zeros": 0, "dd_method": "taylor"},
                  }
    err_dict = {}
    for key, leja_p in leja_plist.items():
        l1_err_list, l2_err_list = [], []
        for n_leja in n_leja_list:
            i, l1_expmv_err, l2_expmv_err = plot_leja_conjugate_ellipse_error(
                    a=leja_p["a"], b=b, c=leja_p["c"], eigJ=eigdtJ, leja_n_zeros=leja_p["leja_n_zeros"],
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

