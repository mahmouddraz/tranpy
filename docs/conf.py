"""Sphinx configuration: Markdown pages (MyST) plus API reference from docstrings."""

from importlib.metadata import version as _version

project = "TranPy"
author = "Mahmoud Draz"
copyright = "2025-2026, Mahmoud Draz"
release = _version("tranpy-ai")

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_heading_anchors = 3
exclude_patterns = ["_build", "README.md", "GITHUB_PAGES_SETUP.md"]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "exclude-members": "set_fit_request, set_score_request, set_predict_request, set_predict_proba_request",
}
autodoc_typehints = "description"
autodoc_mock_imports = ["powerfactory"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "sphinx_rtd_theme"
