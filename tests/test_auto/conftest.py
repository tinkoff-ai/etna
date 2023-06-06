from os import unlink

import pytest
from optuna.storages import RDBStorage
from typing_extensions import Literal
from typing_extensions import NamedTuple

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
        state: Literal["RUNNING", "WAITING", "COMPLETE", "PRUNED", "FAIL"] = "COMPLETE"

    complete_trials = [
        Trial(user_attrs={"pipeline": pipeline.to_dict(), "SMAPE_median": float(i)})
        for i, pipeline in enumerate((Pipeline(NaiveModel(j), horizon=7) for j in range(10)))
    ]
    fail_trials = [Trial(user_attrs={}, state="FAIL")]

    return complete_trials + fail_trials
