#!/bin/bash

# for python3 type hints
pip install sphinx
pip install sphinx-autodoc-typehints

# auto-gen api docs from python src
sphinx-apidoc -f -o source ../ormatex_py ../ormatex_py/progression
