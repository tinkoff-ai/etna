# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Some code in conf.py and `_templates` are used from `github.com/jdb78/pytorch-forecasting/tree/v0.9.2/docs/source` under MIT License
import os
from pathlib import Path
import shutil
import sys

import toml
from sphinx.application import Sphinx
from sphinx.ext.autosummary import Autosummary

SOURCE_PATH = Path(os.path.dirname(__file__))  # noqa # docs source
PROJECT_PATH = SOURCE_PATH.joinpath("../..")  # noqa # project root

COMMIT_SHORT_SHA = os.environ.get("CI_COMMIT_SHORT_SHA", None)
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", None)

sys.path.insert(0, str(PROJECT_PATH))  # noqa

import etna  # isort:skip

# -- Project information -----------------------------------------------------

project = 'ETNA Time Series Library'
copyright = '2021, etna-tech@tinkoff.ru'
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
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx-mathjax-offline",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

intersphinx_mapping = {
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "pytorch_forecasting": ("https://pytorch-forecasting.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/3.5.0/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None)
}

autodoc_typehints = "both"
autodoc_typehints_description_target = "all"
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# setup configuration
def skip(app, what, name, obj, skip, options):
    """
    Document __init__ methods
    """
    if name == "__init__":
        return True
    return skip


apidoc_output_folder = SOURCE_PATH.joinpath("api")

PACKAGES = [etna.__name__]


def get_by_name(string: str):
    """
    Import by name and return imported module/function/class

    Args:
        string (str): module/function/class to import, e.g. 'pandas.read_csv' will return read_csv function as
        defined by pandas

    Returns:
        imported object
    """
    class_name = string.split(".")[-1]
    module_name = ".".join(string.split(".")[:-1])

    if module_name == "":
        return getattr(sys.modules[__name__], class_name)

    mod = __import__(module_name, fromlist=[class_name])
    return getattr(mod, class_name)


class ModuleAutoSummary(Autosummary):
    def get_items(self, names):
        new_names = []
        for name in names:
            mod = sys.modules[name]
            mod_items = getattr(mod, "__all__", mod.__dict__)
            for t in mod_items:
                if "." not in t and not t.startswith("_"):
                    obj = get_by_name(f"{name}.{t}")
                    if hasattr(obj, "__module__"):
                        mod_name = obj.__module__
                        t = f"{mod_name}.{t}"
                    if t.startswith("etna"):
                        new_names.append(t)
        new_items = super().get_items(sorted(new_names, key=lambda x:  x.split(".")[-1]))
        return new_items


def setup(app: Sphinx):
    app.connect("autodoc-skip-member", skip)
    app.add_directive("moduleautosummary", ModuleAutoSummary)
    app.add_js_file("https://buttons.github.io/buttons.js", **{"async": "async"})


autodoc_member_order = "groupwise"
autoclass_content = "both"

# autosummary
autosummary_generate = True
shutil.rmtree(SOURCE_PATH.joinpath("api"), ignore_errors=True)