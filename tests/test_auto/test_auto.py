from functools import partial
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from typing_extensions import Literal

from etna.auto import Auto
from etna.auto.auto import _Callback
from etna.auto.auto import _Initializer
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.models import MovingAverageModel
from etna.models import NaiveModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform


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


@pytest.mark.parametrize("tune_size", [0, 2])
def test_fit_called_tuning_pool(
    tune_size,
    ts=MagicMock(),
    auto=MagicMock(),
    timeout=4,
    n_trials=2,
    initializer=MagicMock(),
    callback=MagicMock(),
):
    auto._get_tuner_timeout = partial(Auto._get_tuner_timeout, self=auto)
    auto._get_tuner_n_trials = partial(Auto._get_tuner_n_trials, self=auto)

    Auto.fit(
        self=auto,
        ts=ts,
        timeout=timeout,
        n_trials=n_trials,
        initializer=initializer,
        callback=callback,
        tune_size=tune_size,
    )

    auto._pool_optuna.tune.assert_called_with(
        objective=auto.objective.return_value, runner=auto.runner, n_trials=n_trials, timeout=timeout
    )


@pytest.mark.parametrize("tune_size", [0, 2])
def test_fit_called_tuning_top_pipelines(
    tune_size,
    ts=MagicMock(),
    auto=MagicMock(),
    timeout=4,
    n_trials=2,
    initializer=MagicMock(),
    callback=MagicMock(),
):
    auto._get_tuner_timeout = partial(Auto._get_tuner_timeout, self=auto)
    auto._get_tuner_n_trials = partial(Auto._get_tuner_n_trials, self=auto)

    Auto.fit(
        self=auto,
        ts=ts,
        timeout=timeout,
        n_trials=n_trials,
        initializer=initializer,
        callback=callback,
        tune_size=tune_size,
    )

    assert auto._fit_tuner.call_count == tune_size


@pytest.mark.parametrize("suppress_logging", [False, True])
@patch("etna.auto.auto.ConfigSampler", return_value=MagicMock())
@patch("etna.auto.auto.Optuna", return_value=MagicMock())
def test_init_optuna(
    optuna_mock,
    sampler_mock,
    suppress_logging,
    auto=MagicMock(),
):
    Auto._init_pool_optuna(self=auto, suppress_logging=suppress_logging)

    optuna_mock.assert_called_once_with(
        direction="maximize", study_name=auto._pool_folder, storage=auto.storage, sampler=sampler_mock.return_value
    )


def test_fit_without_tuning(
    example_tsds,
    optuna_storage,
    pool=(Pipeline(MovingAverageModel(5), horizon=7), Pipeline(NaiveModel(1), horizon=7)),
):
    auto = Auto(
        MAE(),
        pool=pool,
        metric_aggregation="median",
        horizon=7,
        storage=optuna_storage,
    )
    auto.fit(ts=example_tsds, n_trials=2)

    assert len(auto._pool_optuna.study.trials) == 2
    assert len(auto.summary()) == 2
    assert len(auto.top_k(k=5)) == 2
    assert len(auto.top_k(k=1)) == 1
    assert auto.top_k(k=1)[0].to_dict() == pool[0].to_dict()


@pytest.mark.parametrize("tune_size", [1, 2])
def test_fit_with_tuning(
    tune_size,
    example_tsds,
    optuna_storage,
    pool=(
        Pipeline(MovingAverageModel(5), horizon=7),
        Pipeline(NaiveModel(1), horizon=7),
        Pipeline(
            LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=list(range(7, 21)))], horizon=7
        ),
    ),
):
    auto = Auto(
        MAE(),
        pool=pool,
        metric_aggregation="median",
        horizon=7,
        storage=optuna_storage,
    )
    auto.fit(ts=example_tsds, n_trials=11, tune_size=tune_size)

    assert len(auto._pool_optuna.study.trials) == 3
    assert len(auto.summary()) == 11
    assert len(auto.top_k(k=5)) == 5
    assert len(auto.top_k(k=1)) == 1
    assert isinstance(auto.top_k(k=1)[0].model, MovingAverageModel)


def test_summary(
    trials,
    auto=MagicMock(),
):
    pool_trials = trials[:3]
    tune_trials_0 = trials[3:6]
    tune_trials_1 = trials[6:]

    auto._pool_optuna.study.get_trials.return_value = pool_trials
    auto._make_pool_summary = partial(Auto._make_pool_summary, self=auto)  # essential for summary

    tune_0 = MagicMock()
    tune_0.pipeline = pool_trials[0].user_attrs["pipeline"]
    tune_0._init_optuna.return_value.study.get_trials.return_value = tune_trials_0
    tune_1 = MagicMock()
    tune_1.pipeline = pool_trials[1].user_attrs["pipeline"]
    tune_1._init_optuna.return_value.study.get_trials.return_value = tune_trials_1
    auto._init_tuners.return_value = [tune_0, tune_1]
    auto._make_tune_summary = partial(Auto._make_tune_summary, self=auto)  # essential for summary

    df_summary = Auto.summary(self=auto)

    assert len(df_summary) == len(trials)
    expected_smape = pd.Series([trial.user_attrs.get("SMAPE_median") for trial in trials])
    pd.testing.assert_series_equal(df_summary["SMAPE_median"], expected_smape, check_names=False)


@pytest.mark.parametrize("k, expected_k", [(1, 1), (2, 2), (3, 3), (20, 10)])
def test_top_k(
    trials,
    k,
    expected_k,
    auto=MagicMock(),
):
    auto.target_metric.name = "SMAPE"
    auto.metric_aggregation = "median"
    auto.target_metric.greater_is_better = False

    pool_trials = trials[:3]
    tune_trials_0 = trials[3:6]
    tune_trials_1 = trials[6:]

    auto._pool_optuna.study.get_trials.return_value = pool_trials
    auto._make_pool_summary = partial(Auto._make_pool_summary, self=auto)  # essential for summary

    tune_0 = MagicMock()
    tune_0.pipeline = pool_trials[0].user_attrs["pipeline"]
    tune_0._init_optuna.return_value.study.get_trials.return_value = tune_trials_0
    tune_1 = MagicMock()
    tune_1.pipeline = pool_trials[1].user_attrs["pipeline"]
    tune_1._init_optuna.return_value.study.get_trials.return_value = tune_trials_1
    auto._init_tuners.return_value = [tune_0, tune_1]
    auto._make_tune_summary = partial(Auto._make_tune_summary, self=auto)  # essential for summary

    auto._top_k = partial(Auto._top_k, self=auto)
    df_summary = Auto.summary(self=auto)
    auto.summary = MagicMock(return_value=df_summary)

    top_k = Auto.top_k(auto, k=k)

    assert len(top_k) == expected_k
    assert [pipeline.model.lag for pipeline in top_k] == [i for i in range(expected_k)]  # noqa C416


def test_summary_after_fit(
    example_tsds,
    optuna_storage,
    pool=(
        Pipeline(MovingAverageModel(5), horizon=7),
        Pipeline(NaiveModel(1), horizon=7),
        Pipeline(
            LinearPerSegmentModel(), transforms=[LagTransform(in_column="target", lags=list(range(7, 21)))], horizon=7
        ),
    ),
):
    auto = Auto(
        MAE(),
        pool=pool,
        metric_aggregation="median",
        horizon=7,
        storage=optuna_storage,
    )
    auto.fit(ts=example_tsds, n_trials=11, tune_size=2)

    df_summary = auto.summary()
    assert {"hash", "pipeline", "state", "study"}.issubset(set(df_summary.columns))
    assert len(df_summary) == 11
    assert len(df_summary[df_summary["study"] == "pool"]) == 3
    df_summary_tune_0 = df_summary[df_summary["study"] == "tuning/edddb11f9acb86ea0cd5568f13f53874"]
    df_summary_tune_1 = df_summary[df_summary["study"] == "tuning/591e66b111b09cbc351249ff4e214dc8"]
    assert len(df_summary_tune_0) == 4
    assert len(df_summary_tune_1) == 4
    assert isinstance(df_summary_tune_0.iloc[0]["pipeline"].model, LinearPerSegmentModel)
    assert isinstance(df_summary_tune_1.iloc[0]["pipeline"].model, MovingAverageModel)
