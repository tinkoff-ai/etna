#!/usr/bin/env bash

# Adapted from Catalyst Team: https://github.com/catalyst-team/catalyst (Apache-2.0 License)

# Cause the script to exit if a single command fails
set -eo pipefail -v

function setup_env() {
  poetry export -f requirements.txt --without-hashes --extras all --output all-requirements.txt
  pip uninstall -r all-requirements.txt -y
  pip install poetry
  poetry install
}

function require_libs() {
  # base .etna file when no parameters are given
  CONFIG="$(echo "
[etna]
torch_required = false
prophet_required = false
wandb_required = false
")"
  for REQUIRED in "$@"
  do
    CONFIG="$(echo "$CONFIG" | sed -re "s/^($REQUIRED)_required\\s*=\\s*false/\\1_required = true/gi")"
  done
  echo "$CONFIG" > .etna
}

################################  pipeline 00  ################################
setup_env
require_libs

cat <<EOT > .etna
[etna]
torch_required = false
prophet_required = false
wandb_required = false
EOT

python -c """
from etna import models
try:
    models.ProphetModel
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

################################  pipeline 01  ################################
setup_env
require_libs

cat <<EOT > .etna
[etna]
torch_required = false
prophet_required = false
wandb_required = false
EOT

python -c """
from etna import loggers
try:
    loggers.WandbLogger
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

################################  pipeline 02  ################################
setup_env
require_libs prophet

cat <<EOT > .etna
[etna]
torch_required = false
prophet_required = true
wandb_required = false
EOT

python -c """
try:
    from etna import models
    models.ProphetModel
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

poetry install -E prophet

python -c """
from etna.models import ProphetModel
"""

################################  pipeline 03  ################################
# even if Prophet installed, available dependencies should be parsed from
# settings

setup_env
require_libs

cat <<EOT > .etna
[etna]
torch_required = false
prophet_required = false
wandb_required = false
EOT

poetry install -E prophet

python -c """
from etna import models
try:
    models.ProphetModel
except (AttributeError, ImportError, AssertionError):
    pass  # Ok
else:
    raise AssertionError('\'ImportError\' or \'AssertionError\' expected')
"""

################################  pipeline 99  ################################
rm .etna
rm all-requirements.txt