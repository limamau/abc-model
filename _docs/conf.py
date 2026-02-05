# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "abc-model"
copyright = "ABC³ Group"
author = "ABC³ Group"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "../figs/logo.png"
html_theme_options = {
    # "logo_only": True,
    # "display_version": False,
    "repository_url": "https://github.com/limamau/abc-model",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "repository_provider": "github",
}
autosummary_generate = True
autodoc_member_order = "bysource"
typehints_document_rtype = False
