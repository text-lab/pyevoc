
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"/Users/michelangelo/Documents/PYTHON/PyEvoc/pyevoc-0.2.1")
sys.path.insert(0, str(ROOT))

project = "PyEvoc API Documentation"
author = "Michelangelo Misuraca"
copyright = "2026, Michelangelo Misuraca"
release = "0.2.1"
version = "0.2.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "imported-members": False,
}

autodoc_typehints = "description"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_mock_imports = [
    "stanza",
    "fasttext",
    "fasttext_wheel",
    "lingua",
    "emoji",
    "plotly",
    "matplotlib",
    "sklearn",
    "scipy",
    "networkx",
    "openpyxl",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["pyevoc_api.css"]
html_title = "PyEvoc API Documentation"
html_short_title = "PyEvoc API"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

todo_include_todos = False

# Avoid build failure from duplicated symbols re-exported from package __init__.py.
suppress_warnings = [
    "autodoc.duplicate_object",
]
