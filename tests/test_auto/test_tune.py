from functools import partial
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest
from typing_extensions import Literal

from etna.auto import Tune
from etna.auto.auto import _Callback
from etna.auto.auto import _Initializer
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.metrics import MAE
from etna.models import NaiveModel
from etna.models import SimpleExpSmoothingModel
from etna.pipeline import AutoRegressivePipeline
from etna.pipeline import Pipeline
from etna.pipeline.hierarchical_pipeline import HierarchicalPipeline
from etna.reconciliation import BottomUpReconciliator
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform


def test_objective(
    example_tsds,
    target_metric=MAE(),
    metric_aggregation: Literal["mean"] = "mean",
    metrics=[MAE()],
    backtest_params={},
    initializer=MagicMock(spec=_Initializer),
    callback=MagicMock(spec=_Callback),
    pipeline=Pipeline(NaiveModel()),
    params_to_tune={},
):
    trial = MagicMock()
    _objective = Tune.objective(
        ts=example_tsds,
        pipeline=pipeline,
        params_to_tune=params_to_tune,
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


def test_fit_called_tune(
    ts=MagicMock(),
    tune=MagicMock(),
    timeout=4,
    n_trials=2,
    initializer=MagicMock(),
    callback=MagicMock(),
):
    Tune.fit(
        self=tune,
        ts=ts,
        timeout=timeout,
        n_trials=n_trials,
        initializer=initializer,
        callback=callback,
    )

    tune._optuna.tune.assert_called_with(
        objective=tune.objective.return_value, runner=tune.runner, n_trials=n_trials, timeout=timeout
    )


@pytest.mark.parametrize("suppress_logging", [False, True])
@patch("optuna.samplers.TPESampler", return_value=MagicMock())
@patch("etna.auto.auto.Optuna", return_value=MagicMock())
def test_init_optuna(
    optuna_mock,
    sampler_mock,
    suppress_logging,
    auto=MagicMock(),
):
    auto.configure_mock(sampler=sampler_mock)
    Tune._init_optuna(self=auto, suppress_logging=suppress_logging)

    optuna_mock.assert_called_once_with(
        direction="maximize", study_name=auto.experiment_folder, storage=auto.storage, sampler=sampler_mock
    )


@pytest.mark.parametrize(
    "params, model",
    [
        ({"model.smoothing_level": FloatDistribution(low=0.1, high=1)}, SimpleExpSmoothingModel()),
        ({"model.smoothing_level": FloatDistribution(low=0.1, high=1, log=True)}, SimpleExpSmoothingModel()),
        ({"model.smoothing_level": FloatDistribution(low=0.1, high=1, step=0.1)}, SimpleExpSmoothingModel()),
        ({"model.lag": IntDistribution(low=1, high=5)}, NaiveModel()),
        ({"model.lag": IntDistribution(low=1, high=5, log=True)}, NaiveModel()),
        ({"model.lag": IntDistribution(low=1, high=5, step=2)}, NaiveModel()),
        ({"model.lag": CategoricalDistribution((1, 2, 3))}, NaiveModel()),
        ({"model.smoothing_level": FloatDistribution(low=1, high=5)}, SimpleExpSmoothingModel()),
    ],
)
def test_can_handle_distribution_type(example_tsds, optuna_storage, params, model):
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(model, horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_transforms(example_tsds, optuna_storage):
    params = {
        "transforms.0.value": IntDistribution(low=0, high=17),
        "transforms.1.value": IntDistribution(low=0, high=17),
    }
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(
            NaiveModel(),
            [AddConstTransform(in_column="target", value=8), AddConstTransform(in_column="target", value=4)],
            horizon=7,
        )
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_summary(
    trials,
    tune=MagicMock(),
):
    tune._optuna.study.get_trials.return_value = trials
    tune._summary = partial(Tune._summary, self=tune)  # essential for summary
    df_summary = Tune.summary(self=tune)

    assert len(df_summary) == len(trials)
    assert {"hash", "pipeline", "state"}.issubset(set(df_summary.columns))
    expected_smape = pd.Series([trial.user_attrs.get("SMAPE_median") for trial in trials])
    pd.testing.assert_series_equal(df_summary["SMAPE_median"], expected_smape, check_names=False)


@pytest.mark.parametrize("k, expected_k", [(1, 1), (2, 2), (3, 3), (20, 10)])
def test_top_k(
    trials,
    k,
    expected_k,
    tune=MagicMock(),
):
    tune.target_metric.name = "SMAPE"
    tune.metric_aggregation = "median"
    tune.target_metric.greater_is_better = False

    tune._optuna.study.get_trials.return_value = trials
    tune._summary = partial(Tune._summary, self=tune)
    tune._top_k = partial(Tune._top_k, self=tune)

    df_summary = Tune.summary(self=tune)
    tune.summary = MagicMock(return_value=df_summary)

    top_k = Tune.top_k(tune, k=k)

    assert len(top_k) == expected_k
    assert [pipeline.model.lag for pipeline in top_k] == [i for i in range(expected_k)]  # noqa C416


@pytest.mark.parametrize(
    "pipeline",
    [
        (Pipeline(NaiveModel(1), horizon=7)),
        (AutoRegressivePipeline(model=NaiveModel(1), horizon=7, transforms=[])),
        (AutoRegressivePipeline(model=NaiveModel(1), horizon=7, transforms=[DateFlagsTransform()])),
    ],
)
def test_tune_run(example_tsds, optuna_storage, pipeline):
    tune = Tune(
        pipeline=pipeline,
        target_metric=MAE(),
        metric_aggregation="median",
        horizon=7,
        storage=optuna_storage,
    )
    tune.fit(ts=example_tsds, n_trials=2)

    assert len(tune._optuna.study.trials) == 2
    assert len(tune.summary()) == 2
    assert len(tune.top_k()) <= 2
    assert len(tune.top_k(k=1)) == 1


@pytest.mark.parametrize(
    "pipeline, params_to_tune",
    [
        (Pipeline(NaiveModel(1), horizon=7), {"model.lag": IntDistribution(low=1, high=5)}),
    ],
)
def test_tune_run_custom_params_to_tune(example_tsds, optuna_storage, pipeline, params_to_tune):
    tune = Tune(
        pipeline=pipeline,
        params_to_tune=params_to_tune,
        target_metric=MAE(),
        metric_aggregation="median",
        horizon=7,
        storage=optuna_storage,
    )
    tune.fit(ts=example_tsds, n_trials=2)

    assert tune.params_to_tune == params_to_tune
    assert len(tune._optuna.study.trials) == 2
    assert len(tune.summary()) == 2
    assert len(tune.top_k()) == 2
    assert len(tune.top_k(k=1)) == 1


@pytest.mark.parametrize(
    "pipeline",
    [
        (
            HierarchicalPipeline(
                reconciliator=BottomUpReconciliator(target_level="total", source_level="market"),
                model=NaiveModel(1),
                transforms=[],
                horizon=1,
            )
        ),
    ],
)
def test_tune_hierarchical_run(
    market_level_constant_hierarchical_ts,
    optuna_storage,
    pipeline,
):
    tune = Tune(
        pipeline=pipeline,
        target_metric=MAE(),
        metric_aggregation="median",
        horizon=7,
        backtest_params={"n_folds": 2},
        storage=optuna_storage,
    )
    tune.fit(ts=market_level_constant_hierarchical_ts, n_trials=2)

    assert len(tune._optuna.study.trials) == 2
    assert len(tune.summary()) == 2
    assert len(tune.top_k()) <= 2
    assert len(tune.top_k(k=1)) == 1
