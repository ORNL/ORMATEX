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
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
import numpy as np
import json

from padephi_s import padephi_s
from cf_rat_phi import cf_rat_phi


class Pole(StrEnum):
    Real = "real"
    Conj = "conj"


@dataclass
class PFD:
    poles: list[str]
    coeffs: list[str]
    poly: list[str]
    pole_type: Pole


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="write_rational_phi",
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

    json_dict: dict[str, dist] = {}
    json_filename = "%s_poles.json" % method if not args.f else args.f

    filename = "%s_poles.txt" % method if not args.f else args.f

    with open(filename, "w", encoding="utf-8") as file:
        file.write("{\n")

        for degree in range(2, 16+1, 2):
            if method == 'cram':
                method_str = f"{method}_{degree:d}"
                z, c, r_inf = cf_rat_phi(degree, s);

                json_dict[method_str] = asdict(
                    PFD(
                        poles=[f"{zi.real:+1.17e}{zi.imag:+1.17e}j" for zi in z[:len(z) // 2]],
                        coeffs=[f"{ci.real:+1.17e}{ci.imag:+1.17e}j" for ci in c[:len(c) // 2]],
                        poly=[f"{r_inf:+1.17e}"],
                        pole_type=Pole.Conj,
                    )
                )

                file.write(f"'{method_str}':\n")
                file.write("([ ")
                lines = [f"{z_str},\n   " for z_str in json_dict[method_str]["poles"]]
                file.writelines(lines)
                file.write("],\n [ ")
                lines = [f"{c_str},\n   " for c_str in json_dict[method_str]["coeffs"]]
                file.writelines(lines)
                file.write("],\n")
                file.write(f" {json_dict[method_str]['poly'][0]}),\n")

            elif method == 'pade':
                # numerator and denominator degree
                n = degree
                # walk over relevant m, without duplicates
                for m in list(dict.fromkeys([0, degree-2, degree-1, degree])):
                    method_str = f"{method}_{m:d}_{n:d}"

                    r, a, b, z, c, r_inf = padephi_s(m, n, s=s)
                    zc_sorted = sorted(zip(z, c), key=lambda zc: zc[0].imag)
                    z, c = zip(*zc_sorted)

                    json_dict[method_str] = asdict(
                        PFD(
                            poles=[f"{zi.real:+.34e}{zi.imag:+.34e}j" for zi in z[:len(z) // 2]],
                            coeffs=[f"{ci.real:+.34e}{ci.imag:+.34e}j" for ci in c[:len(c) // 2]],
                            poly=[f"{r_inf:+.34e}"],
                            pole_type=Pole.Conj,
                        )
                    )

                    file.write(f"'{method_str}':\n")
                    file.write("([ ")
                    lines = [f"{z_str},\n   " for z_str in json_dict[method_str]["poles"]]
                    file.writelines(lines)
                    file.write("],\n [ ")
                    lines = [f"{c_str},\n   " for c_str in json_dict[method_str]["coeffs"]]
                    file.writelines(lines)
                    file.write("],\n")
                    file.write(f" {json_dict[method_str]['poly'][0]}),\n")

        file.write("}\n")
    print(f"written PFD(phi_{s}) poles and coefficients for {method=} in '{filename}'.")

    with open(json_filename, "w", encoding="utf-8") as file:
        json.dump(json_dict, file, indent=2)
    print(f"written PFD(phi_{s}) poles and coefficients for {method=} in '{json_filename}'.")
