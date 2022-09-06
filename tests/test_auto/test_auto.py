from os import unlink
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from optuna.storages import RDBStorage
from typing_extensions import Literal

from etna.auto import Auto
from etna.auto.auto import _Callback
from etna.auto.auto import _Initializer
from etna.metrics import MAE
from etna.models import NaiveModel
from etna.pipeline import Pipeline


@pytest.fixture()
def optuna_storage():
    yield RDBStorage("sqlite:///test.db")
    unlink("test.db")


def test_objective(
    example_tsds,
    target_metric=MAE(),
    metric_aggregation: Literal["mean"] = "mean",
    metrics=[MAE()],
    backtest_params={},
    initializer=MagicMock(spec=_Initializer),
    callback=MagicMock(spec=_Callback),
    relative_params={
        "_target_": "etna.pipeline.Pipeline",
        "horizon": 7,
        "model": {"_target_": "etna.models.NaiveModel", "lag": 1},
    },
):
    trial = MagicMock(relative_params=relative_params)
    _objective = Auto.objective(
        ts=example_tsds,
        target_metric=target_metric,
        metric_aggregation=metric_aggregation,
        metrics=metrics,
        backtest_params=backtest_params,
        initializer=initializer,
        callback=callback,
    )
    aggregated_metric = _objective(trial)
    assert isinstance(aggregated_metric, float)

    initializer.assert_called_once()
    callback.assert_called_once()


def test_fit(
    ts=MagicMock(),
    auto=MagicMock(),
    timeout=4,
    n_trials=2,
    initializer=MagicMock(),
    callback=MagicMock(),
):

    Auto.fit(
        self=auto,
        ts=ts,
        timeout=timeout,
        n_trials=n_trials,
        initializer=initializer,
        callback=callback,
    )

    auto._optuna.tune.assert_called_with(
        objective=auto.objective.return_value, runner=auto.runner, n_trials=n_trials, timeout=timeout
    )


@patch("etna.auto.auto.ConfigSampler", return_value=MagicMock())
@patch("etna.auto.auto.Optuna", return_value=MagicMock())
def test_init_optuna(
    optuna_mock,
    sampler_mock,
    auto=MagicMock(),
):

    Auto._init_optuna(self=auto)

    optuna_mock.assert_called_once_with(
        direction="minimize", study_name=auto.experiment_folder, storage=auto.storage, sampler=sampler_mock.return_value
    )


def test_simple_auto_run(example_tsds, optuna_storage):

    auto = Auto(
        MAE(),
        pool=[Pipeline(NaiveModel(1), horizon=7), Pipeline(NaiveModel(2), horizon=7)],
        metric_aggregation="percentile_95",
        horizon=7,
        storage=optuna_storage,
    )
    auto.fit(ts=example_tsds, n_trials=2)

    assert len(auto._optuna.study.trials) == 2
    assert len(auto.summary()) == 2
