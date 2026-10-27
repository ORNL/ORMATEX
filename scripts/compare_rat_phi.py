import argparse
import ast
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from padephi_s import padephi_s
from cf_rat_phi import cf_rat_phi, phiS


def plot_error_real(method, pfd_methods, order, pade_sub_str, s=0):
    match method:
        case "pade":
            pade_sub = (
                int(pade_sub_str)
                if not pade_sub_str == "max"
                else order
            )
            nominator_order = order - pade_sub
            method_str = f"{method}_{nominator_order:d}_{order:d}"
        case "cram":
            method_str = f"{method}_{order:d}"
        case _:
            raise NotImplementedError()

    if method_str in pfd_methods:
        z, c, r_inf = pfd_methods[method_str]
        # we only save one pole for each complex conjugate pair, needs factor two
        pfd_factor = 2.0
    else:
        # regenerate complex conjuagte poles from scratch
        pfd_factor = 1.0
        match method:
            case "pade":
                _, _, _, z, c, r_inf = padephi_s(nominator_order, order, s=s)
            case "cram":
                z, c, r_inf = cf_rat_phi(order, s=s)

    z = np.asarray(z, dtype=complex).reshape((-1, 1))
    c = np.asarray(c, dtype=complex).reshape((-1, 1))

    prt_str = (
        f"loaded {method_str} from {file_path}:" if pfd_factor == 2.0
        else f"recomputed {method_str}:"
    )
    print(f"{prt_str}\n{z.T=}\n{c.T=}")

    def eval_phi_pfd(Z, s=s, z_pole=z, c_pole=c, pfd_factor=pfd_factor):
        Z = np.atleast_1d(Z)
        shape = Z.shape
        out = pfd_factor * np.sum(c_pole * np.pow(z_pole, -s) / (Z.reshape(-1) - z_pole), axis=0)
        constant = r_inf if s == 0 else 0.0
        return np.real(constant + out.reshape(shape))

    # plot approximation
    ts = np.linspace(-100, 0.01, 1000)

    exp_ts = np.exp(ts)
    phi1_ts = phiS(ts, s=1)
    phi2_ts = phiS(ts, s=2)

    exp_ts_0 = eval_phi_pfd(ts, s=0)
    phi1_ts_m1 = (exp_ts_0 - 1.0) / ts
    phi1_ts_0 = eval_phi_pfd(ts, s=1)
    exp_ts_1 = 1. + ts * phi1_ts_0
    phi2_ts_0 = eval_phi_pfd(ts, s=2)
    phi1_ts_1 = 1. + ts * phi2_ts_0
    exp_ts_2 = 1. + ts * phi1_ts_1

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 10),
        sharex=True,
    )

    ax1.plot(ts, exp_ts, 'k:', zorder=5, label=r"$\exp(t)$")
    ax1.plot(ts, exp_ts_0, label=r"$r_0(t)$")
    ax1.plot(ts, exp_ts_1, label=r"$1 + t r_1(t)$")
    ax1.plot(ts, exp_ts_2, label=r"$1 + t + t^2 r_2(t)$")
    ax1.legend()
    ax1.set_title("Exponential approximation")

    ax2.plot(ts, phi1_ts, 'k:', zorder=5, label=r"$\varphi_1(t)$")
    ax2.plot(ts, phi1_ts_m1, label=r"$(r_1(t) - 1)/t$")
    ax2.plot(ts, phi1_ts_0, label=r"$r_1(t)$")
    ax2.plot(ts, phi1_ts_1, label=r"$1 + t r_2(t)$")
    ax2.legend()
    ax2.set_title(r"$\varphi_1$ approximation")

    ax3.plot(ts, 1e-20 + np.abs(exp_ts_0 - exp_ts))
    ax3.plot(ts, 1e-20 + np.abs(exp_ts_1 - exp_ts))
    ax3.plot(ts, 1e-20 + np.abs(exp_ts_2 - exp_ts))
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$t$")
    #ax3.legend()
    ax3.set_title("Exponential approximation error")

    ax4.plot(ts, 1e-20 + np.abs(phi1_ts_m1 - phi1_ts))
    ax4.plot(ts, 1e-20 + np.abs(phi1_ts_0 - phi1_ts))
    ax4.plot(ts, 1e-20 + np.abs(phi1_ts_1 - phi1_ts))
    ax4.set_yscale("log")
    ax4.set_xlabel(r"$t$")
    #ax4.legend()
    ax4.set_title(r"$\varphi_1$ approximation error")

    fig.suptitle(f"Errors for method {method_str}")

    plots_dir = Path("phi_error_plots")
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"phi_comparison_{method_str}.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compare_rat_phi",
        description="Compare rational Partial Fraction Decomposition (PFD) approximations.",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="",
        help="The path to the file to be read."
    )
    parser.add_argument(
        "-s",
        type=int,
        default=0,
        help="The phi-function order (default 0, the exponential)."
    )
    parser.add_argument(
        "--method",
        choices=["pade", "cram"],
        default="cram",
        help="PFD methods."
    )
    parser.add_argument(
        "--pade-sub-diag",
        choices=["0", "1", "2", "max"],
        default="0",
        help="Pade subdiagonal index."
    )
    parser.add_argument(
        "--n-pfd",
        type=int,
        default=2,
        help="The number of partial fractions (even, default 2, the lowest for complex poles)."
    )
    args = parser.parse_args()
    s = args.s
    assert(s == 0) # comparison is only valid for s==0 right now
    method = args.method

    filename = "%s_poles.txt" % method if not args.file else args.file
    file_path = Path(filename)
    pfd_methods: dict[str, tuple] = {}
    if file_path.exists():
        text_content = file_path.read_text(encoding="utf-8")
        pfd_methods = ast.literal_eval(text_content)

    order = args.n_pfd
    assert(order == -1 or order > 0 and order % 2 == 0)
    pade_sub_str = args.pade_sub_diag

    if order > 0:
        plot_error_real(method, pfd_methods, order, pade_sub_str, s=s)
    elif order == -1:
        for order_ in range(2, 16+1, 2):
            plot_error_real(method, pfd_methods, order_, pade_sub_str, s=s)
