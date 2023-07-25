# Contribution guide

## How to start?

Contributing is quite easy: suggest ideas and make them done.
We use [GitHub issues](https://github.com/tinkoff-ai/etna/issues) for bug reports and feature requests.

Every good PR usually consists of:
- feature implementation :)
- documentation to describe this feature to other people
- tests to ensure everything is implemented correctly

## Step-by-step guide

### Before the PR
Please ensure that you have read the following docs:
- [documentation](https://etna-docs.netlify.app/)
- [tutorials](https://github.com/tinkoff-ai/etna/tree/master/examples)
- [changelog](https://github.com/tinkoff-ai/etna/blob/master/CHANGELOG.md)

### Setting up your development environment

Before writing any code it is useful to set up a development environment.
1. Clone etna library to some folder and go inside:
```bash
git clone https://github.com/tinkoff-ai/etna.git etna/
cd etna
```
2. Run installation with `poetry` ([poetry installation guide](https://python-poetry.org/docs/#installation)):
```bash
poetry install -E all-dev
```
3. Activate virtual environment created by poetry:
```bash
poetry shell
```

To connect virtual environment interpreter to IDE the `which python` command can be useful.

### Suggesting a feature (optional)

This is an optional step, you could skip it if you want to take some of the existing tasks.

1. Make an issue with your feature description;
2. We shall discuss the design and its implementation details;
3. Once we agree that the plan looks good, go ahead and implement it.

### Doging a task

1. Goto [GitHub issues](https://github.com/tinkoff-ai/etna/issues);
2. Pick an issue and comment on the task that you want to work on this task;
3. If you need more context on a specific issue, please ask, and we will discuss the details.

You can also join our [ETNA Community telegram chat](https://t.me/etna_support) to make it easier to discuss.
Once you finish implementing a feature or bugfix, please send a Pull Request.

If you are not familiar with creating a Pull Request, here are some guides:
- [Creating a pull request](https://help.github.com/articles/creating-a-pull-request/);
- [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

### Tests

Do not forget to check that your code passes the unit tests.
```bash
pytest tests -v
pytest etna -v --doctest-modules
```

ETNA uses the [`black`](https://github.com/psf/black) and [`flake8`](https://github.com/pycqa/flake8) with several plugins 
for coding style checks as well as [`mypy`](https://github.com/python/mypy) for type checks, and you must ensure that your code follows it. 
```bash
make format
```

If any of checks fails, the CI will fail and your Pull Request won't be merged.

### Documentation

ETNA uses [Numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) for formatting docstrings. 
The documentation is written in ReST.
Length of a line inside docstrings block must be limited to 100 characters to fit into Jupyter documentation popups.

During creation of Pull Request make sure that your documentation looks good, check:
1. `Parameters` and `Returns` sections have correct names and types;
2. Sections should be
   1. divided correctly without artefacts,
   2. consistent by meaning with [Numpydoc Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html);
3. Lists are rendered correctly;
4. Listings of code, e.g. variable names, are typed with monospaced font;
5. Mathematical formulas are rendered correctly;
6. Links to external sources are active;
7. References to python objects should be active if library is listed in [`intersphinx_mapping`](https://github.com/tinkoff-ai/etna/blob/master/docs/source/conf.py#L68)

Useful links:
1. [ReST Quickref](https://docutils.sourceforge.io/docs/user/rst/quickref.html)
2. [ReST Roles](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html)
3. [ReST Cross-referencing Python objects](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)
4. [Matplotlib Cheetsheet](https://matplotlib.org/sampledoc/cheatsheet.html)
5. [Sklearn example](https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/linear_model/_ridge.py#L321)

#### Tutorials

If you are going to add a jupyter notebook tutorial:
1. Add the notebook into `examples` folder with its prepended number.
2. Add "launch binder" button to the notebook.
3. Add "Table of contents" for headings of the levels 2 and 3.
4. List extensions that are necessary for this notebook to run.
5. Add imports that are unrelated to the topic of the tutorial at the very top.
6. Add new notebook with its table of contents to the `examples/README.md`
7. Add new notebook with to the `README.md`

#### Building

Before building the documentation you may need to install a pandoc package ([pandoc installation guide](https://pandoc.org/installing.html)):
```bash
# Ubuntu
apt-get update && apt-get install -y pandoc

# Mac OS
brew install pandoc

# Windows
choco install pandoc
```

After that, you can try to build the docs with:
```bash
cd docs
make build-docs
```

You could check the result in your browser by opening `build/html/index.html` file.

If you have some issues with building docs - please make sure that you installed the required packages.
