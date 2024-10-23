#!/usr/bin/env python3
from __future__ import print_function, absolute_import
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
import os
import glob
import sys
pwd_ = os.path.dirname(os.path.abspath(__file__))

print("===================== ORMATEX-PY Setup.py =========================")
print("The setup.py script should be executed from the proj root directory.")
print("pwd: " + pwd_)

setup(
    name='ormatex_py',
    version='0.0.1',
    packages=find_packages(),
    description='Matrix exponential routines and exponential time integrators',
    author='William Gurecky, Konstantin Pieper',
    platforms=["Linux", "Mac OS-X"],
    install_requires=['numpy>=1.8.0', 'scipy>=0.12.0', 'jax', 'equinox', 'matplotlib'],
    package_data={'': ['*.txt']},
    license='TBD',
    author_email='gureckywl@ornl.gov',
    keywords='matrix exponential, time integration, exponential time integration',
    # set primary console script up
    entry_points = {
    }
)
