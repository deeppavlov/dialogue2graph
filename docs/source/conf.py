# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../dialogue2graph"))

project = "Dialogue2Graph"
copyright = "2025, Denis Kuznetsov, Anastasia Voznyuk, Andrey Chirkin, Anna Mikhailova, Maria Molchanova, Yuri Peshkichev"
author = "Denis Kuznetsov, Anastasia Voznyuk, Andrey Chirkin, Anna Mikhailova, Maria Molchanova, Yuri Peshkichev"

# Get the deployment environment
on_github = os.environ.get("GITHUB_ACTIONS") == "true"

# Configure URLs for GitHub Pages
if on_github:
    html_baseurl = "/dialogue2graph/dev/"
    html_context = {
        "display_github": True,
        "github_user": "deeppavlov",
        "github_repo": "dialogue2graph",
        "github_version": "dev",
        "conf_py_path": "/docs/source/",
    }

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": True,
    "special-members": "__call__",
    "member-order": "bysource",
    "exclude-members": "_abc_impl, model_fields, model_computed_fields, model_config",
}

autodoc_typehints = "both"

autoapi_dirs = ["../../dialogue2graph"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
suppress_warnings = ["autoapi.python_import_resolution"]
autoapi_ignore = ["*/cli/*.py"]

napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

extlinks = {
    "github_source_link": (
        "https://github.com/deeppavlov/dialogue2graph/tree/dev/%s",
        None,
    ),
}

# Add these configurations
html_js_files = [
    "scripts/pydata-sphinx-theme.js",
    "scripts/bootstrap.js",
    "scripts/fontawesome.js",
]

# Fix base URL for GitHub Pages
html_baseurl = "/dialogue2graph/dev/"

# Important: Add this to handle static files correctly
html_theme_options = {
    "use_edit_page_button": False,
    "navigation_depth": 3,
    "show_toc_level": 2,
    # Add this to fix static file paths
    "static_page_path": "/dialogue2graph/dev/_static/",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/deeppavlov/dialogue2graph",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# Fix relative URLs for GitHub Pages deployment
html_use_relative_paths = True

# Ensure all static paths are properly prefixed for GitHub Pages
if os.environ.get("GITHUB_ACTIONS") == "true":
    html_static_path_suffix = "/dialogue2graph/dev"


def skip_submodules(app, what, name, obj, skip, options):
    if what == "module" and "." not in name:
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)
