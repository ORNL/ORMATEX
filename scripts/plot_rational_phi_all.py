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
from importlib import resources
import json

from plot_rational_phi import plot_error

if __name__ == "__main__":
    s = 0
    methods = [
        "cram",
        "pade",
    ]
    pfd_methods: dict[str, dict] = {}

    # first, read in the methods from the json files
    pfd_library = resources.files("ormatex_py.pfd_library")
    for method in methods:
        file_path = pfd_library / f"{method}_poles.json"
        if file_path.is_file():
            with file_path.open("r", encoding="utf-8") as file:
                # load the method dict
                pfd_methods.update(json.load(file))

    pade_sub_strs = ["0", "1", "2", "max"]
    for method in methods:
        subs = pade_sub_strs if method == "pade" else [""]
        for sub in subs:
            for order in range(2, 16+1, 2):
                plot_error(method, pfd_methods, order, sub, s=s)
