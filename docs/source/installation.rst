.. _installation:

Installation
============

Installing from PyPi
-------------------------

ETNA is available on `PyPI <https://pypi.org/project/etna>`_, so you can use ``pip`` to install it.

Install default version:

.. code-block:: console

    pip install --upgrade pip
    pip install etna

The default version doesn't contain all the dependencies, because some of them are needed only for specific models, e.g. Prophet, PyTorch.
Available user extensions are the following:

- ``prophet``: adds prophet model,
- ``torch``: adds models based on neural nets,
- ``wandb``: adds wandb logger,
- ``auto``: adds AutoML functionality,
- ``statsforecast``: adds models from `statsforecast <https://nixtla.github.io/statsforecast/>`_,
- ``classiciation``: adds time series classification functionality.

Install extension:

.. code-block:: console

    pip install "etna[extension-name]"

Install all the extensions:

.. code-block:: console

    pip install "etna[all]"

Installing from the repository
------------------------------

ETNA can also be installed from the repository. It allows you to get the latest version of the library or version from a specific commit. It can be done by a command:

.. code-block:: console

    pip install "etna[all]@git+https://github.com/tinkoff-ai/etna.git@master"

where you could replace ``master`` branch with some other identifier.

Installing on computing platforms
---------------------------------

The library can be installed on Google Colab and Kaggle platforms. You could use any of the instructions above for this.

On Google Colab you should don't forget to restart the environment after installing the library.

On Kaggle for versions of the library less than 2.2 you should add ``import numba`` before installation like in `this notebook <https://www.kaggle.com/code/goolmonika/forecasting-using-etna-library-60-lines-catboost>`_.


Installing on Apple M1 (ARM)
-------------------------------------

We are trying to make ETNA work with Apple M1 and other ARM chips.
However due to novelty of these architectures some packages ETNA depends on are going to run into some problems or bugs.

List of known problems:

- `CatBoost installation problem <https://github.com/catboost/catboost/issues/1526#issuecomment-978223384>`_
- `Numba (llvmlite) installation problem <https://github.com/numba/llvmlite/issues/693#issuecomment-909501195>`_

Possible workaround:

- Use ``python>=3.9`` and initialize ``virtualenv``.
- Build CatBoost via instruction in the comment above: you will need ``llvm`` installed via ``brew`` and you should specify paths to ``llvm`` and python binaries in flags correctly to make successful build.
- Install built CatBoost whl in ``virtualenv``.
- Install library: ``LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" pip install etna==<version>``. (``LLVM_CONFIG`` flag may be optional and it could be different a little bit in version spec but you should have 11 or 12 major version)
