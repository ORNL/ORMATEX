*********************
ORMATEX Documentation
*********************

Prerequisites
#############

The Documentation is built using sphinx:

    pip install sphinx

Build the docs
##############

To generate docs run:

    ./sphinx-apidoc-gen.sh && make html

from this docs directory.

View the docs
#############

View the Documentation using your web browser of choice.  From the docs directory run:

    firefox build/html/index.html
