import os
import sys

# Add project root to path so autodoc can find src/
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------
project = 'Scen-O-Con'
copyright = '2025, Scen-O-Con Contributors'
author = 'Scen-O-Con Contributors'

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
html_title = 'Scen-O-Con'
html_static_path = ['_static']

html_theme_options = {
    "source_repository": "https://github.com/your-org/ScenarioApproachTool",
    "source_branch": "master",
    "source_directory": "docs/",
}
