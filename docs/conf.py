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
sys.path.insert(0, os.path.abspath('../sarkas'))
sys.path.insert(0, os.path.abspath('../sarkas/algorithm'))
sys.path.insert(0, os.path.abspath('../sarkas/integrators'))
sys.path.insert(0, os.path.abspath('../sarkas/io'))
sys.path.insert(0, os.path.abspath('../sarkas/plotting'))
sys.path.insert(0, os.path.abspath('../sarkas/potentials'))
sys.path.insert(0, os.path.abspath('../sarkas/simulation'))
sys.path.insert(0, os.path.abspath('../sarkas/thermostats'))
sys.path.insert(0, os.path.abspath('../sarkas/tools'))

intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('http://matplotlib.org/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev/', None),
}

# -- Project information -----------------------------------------------------
project = 'Sarkas'
copyright = '2020, MurilloGroup'
author = 'MurilloGroup'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx_rtd_theme',
    'sphinxcontrib.apidoc',
    # 'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    'sphinx.ext.intersphinx'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = ['yaml', 'fdint', 'numba', 'scipy', 'optparse', 'time',
                        'pyfftw', 'pyfiglet', 'tqdm', 'fmm3dpy']


html_last_updated_fmt = '%b, %d, %Y'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build',
                    'notebooks',
                    'scripts',
                    'tests',
                    'Thumbs.db',
                    '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

import sarkas

# -- APIDoc configuration -----------------------------------------------------
apidoc_module_dir = '../sarkas'
apidoc_output_dir = 'api'
apidoc_excluded_paths = ['*tests*', '*notebooks*']
apidoc_separate_modules = True
