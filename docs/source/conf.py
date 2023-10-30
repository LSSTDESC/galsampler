# Configuration file for the Sphinx documentation builder.
#
"""
"""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("galsampler").version
except DistributionNotFound:
    __version__ = "unknown version"

project = "galsampler"
copyright = "2022, Andrew Hearin"
author = "Andrew Hearin"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [".ipynb_checkpoints/*"]

nbsphinx_prompt_width = "0"  # no prompts in nbsphinx

nbsphinx_kernel_name = "python3"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = []

master_doc = "index"
