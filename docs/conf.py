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
import sphinx_bootstrap_theme
import sarkas

sys.path.insert(0, os.path.abspath('../sarkas'))
sys.path.insert(0, os.path.abspath('../sarkas/time_evolution'))
sys.path.insert(0, os.path.abspath('../sarkas/utilities'))
sys.path.insert(0, os.path.abspath('../sarkas/potentials'))
sys.path.insert(0, os.path.abspath('../sarkas/thermostats'))
sys.path.insert(0, os.path.abspath('../sarkas/tools'))

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/dev/', None),
}

# -- Project information -----------------------------------------------------
project = 'Sarkas'
author = 'MurilloGroup'
copyright = '2019-2022, ' + author

# The full version, including alpha/beta/rc tags
release = sarkas.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'recommonmark',
    'sphinx_panels',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

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
latex_engine = 'xelatex'
latex_elements = {'preamble': r'\usepackage{physics}'}
# latex_additional_files = ["physics.sty"]
bibtex_bibfiles = ['references.bib','credits/publications.bib']
bibtex_reference_style = 'author_year'

autodoc_mock_imports = ['yaml', 'numba', 'scipy', 'optparse', 'time',
                        'pyfftw', 'pyfiglet', 'tqdm', 'fmm3dpy']


html_last_updated_fmt = '%b %d, %Y'
html_logo = os.path.join('graphics', os.path.join('logo','logo_s_orange.png'))

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build',
                    '**.ipynb_checkpoints',
                    'notebooks',
                    'scripts',
                    'html',
                    'tests',
                    'Thumbs.db',
                    '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
# html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "ARKAS",
    
    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing with "" (default) or the name of a valid theme such
    # as "amelia" or "cosmo".
    #
    # Note that this is served off CDN, so won't be available offline.
    'bootswatch_theme': "flatly",
    
    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    'bootstrap_version': "3",
    
    'body_max_width': '100%',
    
    # Render the next and previous page links in navbar. (Default: true)
    'navbar_sidebarrel': False,

    
    'navbar_links': [
        ("Get Started", "documentation/get_started"),
        ("Examples","examples/examples"),
        ("Code Development", "code_development/code_dev"),
        ("API", "api/api"),
        ("Credits","credits/credits"),
    ],
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css", 
    "https://fonts.googleapis.com/css2?family=RocknRoll+One&display=swap",
    "my-style.css"
    ]
# panels_add_fontawesome_latex = True

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()


html_js_files = [
    "js/myscript.js",
]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'documentation/*': ['custom_localtoc.html'],
    'code_development/*': ['custom_localtoc.html'],
    'theory/*': ['custom_localtoc.html'],
    }

# -- APIDoc configuration -----------------------------------------------------

# Generate the API documentation when building
autosummary_generate = True
autosummary_generate_overwrite = True
# # A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['sarkas.']

# Make sure the target is unique
autosectionlabel_prefix_document = True
