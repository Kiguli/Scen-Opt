import os
import sys

# Add project root to path so autodoc can find src/
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------
project = 'Scen-Opt'
copyright = '2025, Scen-Opt Contributors'
author = 'Scen-Opt Contributors'

# -- General configuration ---------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_design',
]

# Mock heavy dependencies so docs build without installing solvers
autodoc_mock_imports = [
    'cvxpy',
    'numpy',
    'scipy',
    'jax',
    'pandas',
    'pyarrow',
    'openpyxl',
    'mosek',
    'clarabel',
    'osqp',
    'scs',
    'concurrent',
]

# Napoleon settings for Google/NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------
html_theme = 'furo'
html_title = ' '
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
    "source_repository": "https://github.com/Kiguli/Scen-Opt",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#448AFF",
        "color-brand-content": "#448AFF",
    },
}
