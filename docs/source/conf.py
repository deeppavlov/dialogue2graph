# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../dialogue2graph'))

project = 'Chatsky LLM-Autoconfig'
copyright = '2024, Denis Kuznetsov, Anastasia Voznyuk, Andrey Chirkin'
author = 'Denis Kuznetsov, Anastasia Voznyuk, Andrey Chirkin'

# Get the deployment environment
on_github = os.environ.get("GITHUB_ACTIONS") == "true"

# Configure URLs for GitHub Pages
if on_github:
    html_baseurl = "/chatsky-llm-autoconfig/dev/"
    html_context = {
        "display_github": True,
        "github_user": "deeppavlov",
        "github_repo": "chatsky-llm-autoconfig",
        "github_version": "dev",
        "conf_py_path": "/docs/source/",
    }

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon',
    "sphinx.ext.extlinks",
    'sphinx_autodoc_typehints',
    'sphinxcontrib.apidoc',
]

templates_path = ['_templates']

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__call__",
    "member-order": "bysource",
    "exclude-members": "_abc_impl, model_fields, model_computed_fields, model_config",
}

apidoc_module_dir = '../../dialogue2graph'
apidoc_output_dir = 'reference'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

extlinks = {
    'github_source_link': ("https://github.com/deeppavlov/chatsky-llm-autoconfig/tree/dev/%s", None),
}

# Add these configurations
html_theme_options = {
    "use_edit_page_button": False,
    "navigation_depth": 3,
    "show_toc_level": 2,
}

# Ensure proper path handling
html_use_relative_paths = True
