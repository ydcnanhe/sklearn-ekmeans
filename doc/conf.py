import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as get_version

project = "sklekmeans"
author = "Yudong He"
copyright = f"{datetime.now():%Y}, {author}"
try:
    release = get_version("sklekmeans")
except Exception:
    # Fallback when building docs without installed package (e.g., clean CI)
    release = os.environ.get("SKLEKMEANS_VERSION", "0.0.0")
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_prompt",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# Ensure project root and package are importable during docs build
DOC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(DOC_DIR, os.pardir))
PKG_DIR = os.path.join(ROOT_DIR, "sklekmeans")
for p in [ROOT_DIR, PKG_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_style = "css/sklekmeans.css"
# Set a logo if present; otherwise skip to avoid warnings
_logo_path = os.path.join(DOC_DIR, "_static", "img", "logo.png")
html_logo = "_static/img/logo.png" if os.path.exists(_logo_path) else None
html_css_files = ["css/sklekmeans.css"]
html_sidebars = {
    "quick_start": [],
    "user_guide": [],
    "auto_examples/index": [],
}

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/ydcnanhe/sklearn-ekmeans",
    # "twitter_url": "https://twitter.com/pandas_dev",
    "use_edit_page_button": True,
    "show_toc_level": 1,
    # "navbar_align": "right",  # For testing that the navbar items align properly
}

html_context = {
    "github_user": "ydcnanhe",
    "github_repo": "sklearn-ekmeans",
    "github_version": "main",
    "doc_path": "doc",
}

# -- Options for autodoc ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

sphinx_gallery_conf = {
    # Disable code link embedding by leaving doc_module empty
    "doc_module": (),
    "backreferences_dir": os.path.join("generated"),
    "examples_dirs": ["../examples"],  # can be a list
    "gallery_dirs": ["auto_examples"],
    # No external reference URLs for code linking
    "reference_url": {},
    "remove_config_comments": True,
    # Ensure examples run and are ordered by file name (so numeric prefixes work)
    "filename_pattern": r".*",
    "within_subsection_order": "FileNameSortKey",
}
