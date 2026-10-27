from padephi_s import padephi_s
from cf_rat_phi import cf_rat_phi

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_rat_phi",
        description="Write rational Partial Fraction Decomposition (PFD) tables.",
    )
    parser.add_argument(
        "-f",
        type=str,
        default="",
        help="The path to the file to be written."
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
        default="pade",
        help="PFD methods."
    )
    args = parser.parse_args()
    s = args.s
    method = args.method

    filename = "%s_poles.txt" % method if not args.f else args.f

    with open(filename, "w", encoding="utf-8") as file:
        file.write("{\n")

        for degree in range(2, 16+1, 2):
            if method == 'cram':
                z, c, r_inf = cf_rat_phi(degree, s);
                file.write("'cram_%d':\n" % degree)

                file.write("([ ")
                lines = []
                for k in range(len(z) // 2):
                    lines.append("% 1.17e%+1.17ej,\n   " % (z[k].real, z[k].imag))

                file.writelines(lines)
                file.write("],\n [ ")
                lines = []
                for k in range(len(c) // 2):
                    lines.append("% 1.17e%+1.17ej,\n   " % (c[k].real, c[k].imag))

                file.writelines(lines)
                file.write("],\n")
                file.write(" % 1.17e),\n" % r_inf.real)

            elif method == 'pade':
                # numerator and denominator degree
                n = degree
                # walk over relevant m, without duplicates
                for m in list(dict.fromkeys([0, degree-2, degree-1, degree])):

                    r, a, b, z, c, r_inf = padephi_s(m, n, s=s)
                    zc_sorted = sorted(zip(z, c), key=lambda zc: zc[0].imag)
                    z, c = zip(*zc_sorted)
                    file.write("'pade_%d_%d':\n" % (m, n))

                    file.write("([ ")
                    lines = []
                    for k in range(len(z) // 2):
                        lines.append("% 1.20e%+1.20ej,\n   " % (z[k].real, z[k].imag))

                    file.writelines(lines)
                    file.write("],\n [ ")
                    lines = []
                    for k in range(len(c) // 2):
                        lines.append("% 1.20e%+1.20ej,\n   " % (c[k].real, c[k].imag))

                    file.writelines(lines)
                    file.write("],\n")
                    file.write(" % 1.20e),\n" % r_inf.real)

        file.write("}\n")
    print(f"written PFD(phi_{s}) poles and coefficients for {method=} in '{filename}'.")
