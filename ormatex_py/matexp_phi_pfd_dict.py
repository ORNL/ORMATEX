##############################################################################
# Copyright© 2025 UT-Battelle, LLC
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
import json
import warnings
from pathlib import Path
from importlib import resources

r"""
Dictionary of poles and coefficients for partial fraction expansion based
rational approximation methods for computing

.. math::

    \phi_k(A)*v \approx r_inf*v + \sum_{i} c_i * z_i^{-k} * (A - z_i)^{-1}v.

It is build from files in 'ormatex_py.pfd_library' and explicit overrides.
"""
pfd_dict: dict[str, tuple[list[complex | float], list[complex | float], float]] = {}

# first, read in the methods from the json files
_pfd_library = resources.files("ormatex_py.pfd_library")
_pfd_methods = [
    "cram",
    "pade",
]
for method in _pfd_methods:
    _file_path = _pfd_library / f"{method}_poles.json"
    if _file_path.is_file():
        with _file_path.open("r", encoding="utf-8") as file:
            # load and convert the method dict
            _pfd_imported_dict = json.load(file)

            for method_str, pfd_method in _pfd_imported_dict.items():
                _z = [complex(z_str) for z_str in pfd_method["poles"]]
                _c = [complex(c_str) for c_str in pfd_method["coeffs"]]
                _r_inf = float(pfd_method["poly"][0])
                # not used:
                # _pfd_conj = pfd_method["pole_type"] == "conj"
                # add the converted method to the pfd_dict
                pfd_dict[method_str] = (_z, _c, _r_inf)
    else:
        warnings.warn(f"file '{_file_path}' was not found for ormatex_py; "
                      "not all PFD methods may be available.")

_pfd_dict_overrides: dict[str, tuple[list[complex | float], list[complex | float], float]] = {
'IE': ([1.0], [-0.5], 0.0),
'CN': ([2.0], [-2.0], -1.0),
'cram_16': (
    [ -10.843917078696988026 + 19.277446167181652284j,
      -5.2649713434426468895 + 16.220221473167927305j,
       5.9481522689511774808 + 3.5874573620183222829j,
       3.5091036084149180974 + 8.4361989858843750826j,
       6.4161776990994341923 + 1.1941223933701386874j,
       1.4193758971856659786 + 10.925363484496722585j,
       4.9931747377179963991 + 5.9968817136039422260j,
      -1.4139284624888862114 + 13.497725698892745389j,
     ],
    [ -0.0000005090152186522491565 - 0.00002422001765285228797j,
         0.00021151742182466030907 +  0.0043892969647380673918j,
             113.39775178483930527 +      101.9472170421585645j,
             15.059585270023467528 -     5.7514052776421819979j,
            -64.500878025539646595 -     224.59440762652096056j,
            -1.4793007113557999718 +     1.7686588323782937906j,
            -62.518392463207918892 -      11.19039109428322848j,
           0.041023136835410021273 -    0.15743466173455468191j,
     ],
    2.1248537104952237488e-16 ),
}
# add the overrides after the import
pfd_dict.update(_pfd_dict_overrides)
