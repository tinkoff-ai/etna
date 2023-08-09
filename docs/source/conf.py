# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
from pathlib import Path
import sys

import toml

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

SOURCE_PATH = Path(os.path.dirname(__file__))  # noqa # docs source
PROJECT_PATH = SOURCE_PATH.joinpath("../..")  # noqa # project root

COMMIT_SHORT_SHA = os.environ.get("CI_COMMIT_SHORT_SHA", None)
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", None)

sys.path.insert(0, str(PROJECT_PATH))  # noqa

import etna  # isort:skip

# -- Project information -----------------------------------------------------

project = 'ETNA Time Series Library'
copyright = '2021 - present, etna-tech@tinkoff.ru (Apache 2.0 License)'
author = 'etna-tech@tinkoff.ru'

# The full version, including alpha/beta/rc tags

with open(PROJECT_PATH / "pyproject.toml", "r") as f:
    pyproject_toml = toml.load(f)

if WORKFLOW_NAME == "Publish":
    release = pyproject_toml["tool"]["poetry"]["version"]
else:
    release = f"{COMMIT_SHORT_SHA}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",   # integration of notebooks
    "myst_parser",  # markdown support
    "sphinx_design",  # for ready styling blocks
    "sphinx.ext.napoleon",  # support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # include documentation from docstrings
    "sphinx.ext.autosummary",  # generate autodoc summaries
    "sphinx.ext.doctest",  # test snippets in the documentation¶
    "sphinx.ext.intersphinx",  # link to other projects’ documentation
    "sphinx.ext.mathjax",  # render math via JavaScript
    "sphinx.ext.githubpages",  # creates .nojekyll file
    "sphinx.ext.linkcode",  # add external links to source code
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/.ipynb_checkpoints"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# A boolean that decides whether module names are prepended to all object names.
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_logo = "_static/etna_logo.png"
html_favicon = "_static/etna_favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tinkoff-ai/etna",
            "icon": "fab fa-github",
        },
        {
            "name": "Telegram",
            "url": "https://t.me/etna_support",
            "icon": "fab fa-telegram",
        },
    ],
    "show_prev_next": False,
}

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scikit-learn": ("http://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "pytorch_forecasting": ("https://pytorch-forecasting.readthedocs.io/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/v2.10.1/", None)
}

# -- Options for autodoc extension -------------------------------------------
# Reference: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autoclass_content = "both"
autodoc_typehints = "both"
autodoc_typehints_description_target = "all"

autodoc_default_options = {
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "groupwise",
    "special-members": "__call__",
}

apidoc_output_folder = SOURCE_PATH.joinpath("api")

# -- Options for autosummary extension ---------------------------------------
# Reference: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

# Activates search for `autosummary` directive
autosummary_generate = True

# -- Options for nbsphinx extension -----------------------------------------
# Reference: https://nbsphinx.readthedocs.io/en/latest/configuration.html

notebook_file = "{{ env.doc2path( env.docname, base=None).split('/')[-1] }}"
notebook_url = (
    f"https://github.com/tinkoff-ai/etna/tree/{release}/examples/{notebook_file}"  # noqa
)

nbsphinx_prolog_css_fix = """
.. raw:: html

    <style>
        table {
            margin-left: 0;
            width: auto;
        }
    </style>
"""

nbsphinx_prolog_notebook_link = f"""
View Jupyter notebook on the `GitHub <{notebook_url}>`_.
"""

nbsphinx_prolog = nbsphinx_prolog_css_fix + nbsphinx_prolog_notebook_link


# -- Options for linkcode extension ------------------------------------------
# Reference: https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html


def linkcode_resolve(domain, info):
    """Return URL to source code corresponding.

    The solution is based on code from sktime: https://github.com/sktime/sktime/blob/v0.20.1/docs/source/conf.py#L137
    """

    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(etna.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "etna/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return "https://github.com/tinkoff-ai/etna/blob/{}/{}".format(
        release,
        filename,
    )
