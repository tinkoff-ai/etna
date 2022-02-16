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
- [changelog](https://github.com/tinkoff-ai/etna/blob/master/CHANGELOG.md)
- [documentation](https://etna-docs.netlify.app/)
- [tutorials](https://github.com/tinkoff-ai/etna/tree/master/examples)

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

### New feature

1. Make an issue with your feature description;
2. We shall discuss the design and its implementation details;
3. Once we agree that the plan looks good, go ahead and implement it.

### Bugfix

1. Goto [GitHub issues](https://github.com/tinkoff-ai/etna/issues);
2. Pick an issue and comment on the task that you want to work on this feature;
3. If you need more context on a specific issue, please ask, and we will discuss the details.

You can also join our [ETNA Community telegram chat](https://t.me/etna_support) to make it easier to discuss.
Once you finish implementing a feature or bugfix, please send a Pull Request.

If you are not familiar with creating a Pull Request, here are some guides:
- [Creating a pull request](https://help.github.com/articles/creating-a-pull-request/);
- [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

### Tests

Do not forget to check that your code passes the unit tests:
```bash
poetry install -E tests

pytest tests -v
pytest etna -v --doctest-modules
```
And code style checks:
```bash
poetry install -E style

make format
```

### Documentation

ETNA uses [Numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) for formatting docstrings.
Length of a line inside docstrings block must be limited to 100 characters to fit into Jupyter documentation popups.

You could check the docs with:
```bash
cp examples/*.ipynb docs/source/tutorials
cd docs
make clean
make html
```

Now you could open them into your browser, for example with
```bash
open ./build/html/index.html
```

If you have some issues with building docs - please make sure that you installed the required packages.

```bash
poetry install -E docs
```
You also may need to install pandoc package ([pandoc installation guide](https://pandoc.org/installing.html)):
```bash
# Ubuntu
apt-get update && apt-get install -y pandoc

# Mac OS
brew install pandoc

#Windows
choco install pandoc
```
