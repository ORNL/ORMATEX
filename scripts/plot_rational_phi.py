##############################################################################
# Copyright© 2026 UT-Battelle, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from padephi_s import padephi_s
from cf_rat_phi import cf_rat_phi, phiS


def plot_error(method, pfd_methods, order, pade_sub_str, s=0):
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
        pfd_method = pfd_methods[method_str]
        z = [complex(z_str) for z_str in pfd_method["poles"]]
        c = [complex(c_str) for c_str in pfd_method["coeffs"]]
        r_inf = float(pfd_method["poly"][0])
        # we only save one pole for each complex conjugate pair, needs factor two
        pfd_conj = pfd_method["pole_type"] == "conj"
    else:
        # regenerate complex conjuagte poles from scratch
        pfd_conj = False
        match method:
            case "pade":
                _, _, _, z, c, r_inf = padephi_s(nominator_order, order, s=s)
            case "cram":
                z, c, r_inf = cf_rat_phi(order, s=s)

    z = np.asarray(z, dtype=complex).reshape((-1, 1))
    c = np.asarray(c, dtype=complex).reshape((-1, 1))
    r_inf = float(r_inf)

    prt_str = (
        f"loaded {method_str}:" if pfd_conj
        else f"recomputed {method_str}:"
    )
    print(f"{prt_str}\n{z=}\n{c=}")

    def eval_phi_pfd(Z, s=s, z_pole=z, c_pole=c, r_inf=r_inf, pfd_conj=pfd_conj):
        Z = np.atleast_1d(Z)
        shape = Z.shape
        # initial expression, valid for complex poles, with conjugate pairs
        out = np.sum(
            c_pole * np.pow(z_pole, -s) / (Z.reshape(-1) - z_pole),
            axis=0,
        )
        if np.iscomplexobj(Z):
            if pfd_conj:
                # add the expression with conjugate (c, z)
                out += np.sum(
                    c_pole.conj() * np.pow(z_pole.conj(), -s) / (Z.reshape(-1) - z_pole.conj()),
                    axis=0,
                )
        else:
            out = np.real(out) if not pfd_conj else np.real(2 * out)
        if s == 0:
            out += r_inf
        out = out.reshape(shape)
        return out

    # plot approximation on negative real line
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
        figsize=(12, 10),
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


    # plot approximation on complex plane
    rs = np.linspace(-30, 10, 200)
    cs = np.linspace(-20, 20, 200)
    Rs, Cs = np.meshgrid(rs, cs)
    Zs = Rs + 1.0j*Cs

    exp_zs = np.exp(Zs)
    phi1_zs = phiS(Zs, s=1)

    exp_zs_0 = eval_phi_pfd(Zs, s=0)
    phi1_zs_0 = eval_phi_pfd(Zs, s=1)

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 6),
        sharey=True,
    )

    exp_err = np.log10(1e-20 + np.abs(exp_zs_0 - exp_zs))
    contour_plot_1 = ax1.contour(
        Rs,
        Cs,
        exp_err,
        linestyles="solid",
    )
    cbar = fig.colorbar(contour_plot_1, ax=ax1)
    ax1.scatter(
        [z.real, z.real],
        [z.imag, -z.imag],
        marker="x",
        color="black",
        label=r"poles",
    )
    ax1.set_xlabel(r"$\Re(z)$")
    ax1.set_ylabel(r"$\Im(z)$")
    ax1.legend()
    ax1.set_title(r"Exponential approximation error ($\log_{10}$)")

    phi1_err = np.log10(1e-20 + np.abs(phi1_zs_0 - phi1_zs))
    contour_plot_2 = ax2.contour(
        Rs,
        Cs,
        phi1_err,
        linestyles="solid",
        #label=r"log-error of $\varphi_1(z)$"
    )
    cbar = fig.colorbar(contour_plot_2, ax=ax2)
    ax2.scatter(
        [z.real, z.real],
        [z.imag, -z.imag],
        marker="x",
        color="black",
        label=r"poles",
    )
    ax2.set_xlabel(r"$\Re(z)$")
    ax2.legend()
    ax2.set_title(r"$\varphi_1$ approximation error ($\log_{10}$)")

    fig.suptitle(f"Errors on complex plane for method {method_str}")

    plots_dir = Path("phi_error_plots")
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"phi_error_{method_str}.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="plot_rational_phi",
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
        default=-1,
        help=(
            "The number of partial fractions, counting conjugate pairs. "
            "A value of -1 will run through all available orders (default -1)."
        )
    )
    args = parser.parse_args()
    s = args.s
    assert(s == 0) # comparison is only valid for s==0 right now
    method = args.method

    filename = f"{method}_poles.json" if not args.file else args.file
    file_path = Path(filename)
    pfd_methods: dict[str, dict] = {}
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            pfd_methods = json.load(file)

    order = args.n_pfd
    assert(order == -1 or order > 0 and order % 2 == 0)
    pade_sub_str = args.pade_sub_diag

    if order > 0:
        plot_error(method, pfd_methods, order, pade_sub_str, s=s)
    elif order == -1:
        for order_ in range(2, 16+1, 2):
            plot_error(method, pfd_methods, order_, pade_sub_str, s=s)
