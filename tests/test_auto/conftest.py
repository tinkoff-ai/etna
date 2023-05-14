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
def trials_auto():
    class Trial(NamedTuple):
        user_attrs: dict
        state: Literal["COMPLETE", "RUNNING", "PENDING"] = "COMPLETE"

    return [
        Trial(user_attrs={"pipeline": pipeline.to_dict(), "SMAPE_median": i})
        for i, pipeline in enumerate((Pipeline(NaiveModel(j), horizon=7) for j in range(10)))
    ]


@pytest.fixture()
def trials_tune():
    class Trial(NamedTuple):
        params: dict
        state: Literal["COMPLETE", "RUNNING", "PENDING"] = "COMPLETE"

    return [
        Trial(params={"pipeline": pipeline.to_dict(), "SMAPE_median": i})
        for i, pipeline in enumerate((Pipeline(NaiveModel(j), horizon=7) for j in range(10)))
    ]

