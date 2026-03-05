import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Sparling"
author = "Kavi Gupta"
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "exclude-members": "forward",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]
