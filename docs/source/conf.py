# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
py_src_path = os.path.abspath('../../ormatex_py/')
print("Python proj src path: %s" % py_src_path)
sys.path.insert(0, py_src_path)


# -- Project information -----------------------------------------------------

project = 'ORMATEX'
copyright = '2025, UT Battelle LLC'
author = 'W. Gurecky, K. Pieper'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinx_autodoc_typehints',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

master_doc = 'index'

autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
#html_theme = 'traditional'

html_title = 'ORMATEX'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_sidebars = {
#     '**': [
#         'about.html',
#         'navigation.html',
#         'relations.html',
#         'searchbox.html',
#     ]
# }
#html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# -- Options for Latex output -------------------------------------------------
# author_tex = r'William Gurecky \and Konstantin Pieper'
# latex_documents = [
#     (master_doc, 'ormatex.tex', 'ORMATEX User Guide', author_tex, 'manual')
# ]


latex_elements = {
    'extraclassoptions': 'openany,oneside',
    'preamble': r'\usepackage{enumitem}\setlistdepth{99}',
    "maketitle": "\\input{ml_psa_title.tex}"
    # 'maketitle': maketitle_tex
}

