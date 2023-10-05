# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

import sarkas

sys.path.insert(0, os.path.abspath("../sarkas"))
sys.path.insert(0, os.path.abspath("../sarkas/time_evolution"))
sys.path.insert(0, os.path.abspath("../sarkas/utilities"))
sys.path.insert(0, os.path.abspath("../sarkas/potentials"))
sys.path.insert(0, os.path.abspath("../sarkas/tools"))

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}

# -- Project information -----------------------------------------------------
project = "Sarkas"
author = "MurilloGroup"
copyright = "2019-2022, " + author

# The full version, including alpha/beta/rc tags
release = sarkas.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Parse docstrings written per the NumPy or Google conventions
    "sphinx.ext.autodoc",  # Automatically generate the API reference documentation
    "sphinx.ext.autosummary",  # To recursively extract docstrings of all submodules and their functions
    "sphinx_autodoc_typehints",  # Type hints support for the Sphinx autodoc extension
    "sphinx.ext.autosectionlabel",  # Allows to refer sections its title.
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinxcontrib.bibtex",  # Allows BibTeX citations to be inserted into documentation
    "sphinx.ext.intersphinx",  # To reference parts of other Sphinx documentations
    "nbsphinx",  # Provides a source parser for *.ipynb files
    "recommonmark",  # A markdown parser for docutils --- should be substitued with MySt Markdown
    "sphinx_design",  # For creating cards, grid layout, drop-downs, tabs
    "sphinxext.opengraph",  # to add Open Graph metadata
    "sphinx_copybutton",  # Add a "copy" button to code blocks
    # "sphinx_codeautolink" #Automatic links from code examples to reference documentation --- it does not work properly at the moment
    # "myst_nb",
]
# MyST configuration
# myst_enable_extensions = [
#     "amsmath",
#     "colon_fence",
#     "deflist",
#     "dollarmath",
#     "html_image",
# ]
# myst_url_schemes = ("http", "https", "mailto")
nb_execution_mode = "off"
nbsphinx_allow_errors=True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = False


# # Equation Numbering
# mathjax_config = {
#     'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
# }
# To ensure LaTeX packages are read
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# _PREAMBLE = r'''
# \usepackage{physics}
# '''
# LaTeX configuration
latex_engine = "xelatex"
latex_elements = {"preamble": r"\usepackage{physics}"}
# latex_additional_files = ["physics.sty"]
bibtex_bibfiles = ["references.bib", "credits/publications.bib"]
bibtex_reference_style = "author_year"

autodoc_mock_imports = ["yaml", "numba", "scipy", "optparse", "time", "pyfftw", "pyfiglet", "tqdm", "fmm3dpy"]


html_last_updated_fmt = "%b %d, %Y"
html_logo = os.path.join("graphics", os.path.join("logo", "logo_s_orange.png"))

# source_suffix = {
#     ".rst": "restructuredtext",
#     ".md": "markdown",
# }
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "notebooks", "scripts", "html", "tests", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_last_updated_fmt = "%b %d, %Y"
html_logo = os.path.join("graphics", os.path.join("logo", "Sarkas_v1_for_dark_bg.svg"))

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_theme_options = {
    "use_repository_button": True,
    "repository_url": "https://github.com/murillo-group/sarkas/",
    "home_page_in_toc": True,
    "logo_only": True,
    "show_navbar_depth": 1,
    "use_download_button": True,
    # "extra_navbar": "<p>Your HTML</p>",
    # sidebar properties below
    "toc_title": "Content",
    "show_toc_level": 3,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://fonts.googleapis.com/css2?family=RocknRoll+One&display=swap"
]

# panels_add_fontawesome_latex = True

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# Suffix to be appended to source links, unless they have this suffix already. Default is .txt
html_sourcelink_suffix = ""

# html_js_files = [
#     "js/myscript.js",
# ]


# Open Graph metadata (social media link previews which improve SEO)
ogp_site_url = "https://murillo-group.github.io/sarkas/"
ogp_site_name = "SARKAS: Python MD code for plasma physics"
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta property="twitter:card" content="summary">',
    '<meta property="twitter:title" content="SARKAS: Python MD code for plasma physics">',
    '<meta property="twitter:description" content="A fast pure-Python molecular dynamics suite for non-ideal plasmas.">',
]

# -- APIDoc configuration -----------------------------------------------------

# Generate the API documentation when building
autosummary_generate = True
autosummary_generate_overwrite = True
# # A list of ignored prefixes for module index sorting.
modindex_common_prefix = ["sarkas."]

# Make sure the target is unique
autosectionlabel_prefix_document = True
