Installation guide for Apple M1 (ARM)
=====================================

We are trying to make ETNA work with Apple M1 and other ARM chips.
However due to novelty of these architectures some packages ETNA depends on are going to run into some problems or bugs.

List of known problems:

- `CatBoost installation problem <https://github.com/catboost/catboost/issues/1526#issuecomment-978223384>`_
- `Numba (llvmlite) installation problem <https://github.com/numba/llvmlite/issues/693#issuecomment-909501195>`_

Possible workaround:

- Initialize virtualenv.
- Build CatBoost via instruction in the comment above: you will need llvm installed via brew and you should specify paths to llvm and python binaries in flags correctly to make successful build. 
- Install builded CatBoost whl in virtualenv.
- Install library: ``LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" pip install etna==<version>``. (``LLVM_CONFIG`` could be different a little bit in version spec but you should have 11 or 12 major version)