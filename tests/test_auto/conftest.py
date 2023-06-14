from os import unlink

import pytest
from optuna.storages import RDBStorage
from optuna.structs import TrialState
from typing_extensions import NamedTuple

from etna.auto.utils import config_hash
from etna.models import NaiveModel
from etna.pipeline import Pipeline


@pytest.fixture()
def optuna_storage():
    yield RDBStorage("sqlite:///test.db")
    unlink("test.db")


@pytest.fixture()
def trials():
    class Trial(NamedTuple):
        user_attrs: dict
        state: TrialState = TrialState.COMPLETE

    complete_trials = [
        Trial(
            user_attrs={
                "pipeline": pipeline.to_dict(),
                "SMAPE_median": float(i),
                "hash": config_hash(pipeline.to_dict()),
            }
        )
        for i, pipeline in enumerate((Pipeline(NaiveModel(j), horizon=7) for j in range(10)))
    ]
    fail_trials = [Trial(user_attrs={}, state=TrialState.FAIL)]

    return complete_trials + complete_trials[:3] + fail_trials
