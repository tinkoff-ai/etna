from os import unlink
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.storages import RDBStorage
from typing_extensions import Literal
from typing_extensions import NamedTuple

from etna.auto import Tune
from etna.auto.auto import AutoBase
from etna.auto.auto import _Callback
from etna.auto.auto import _Initializer
from etna.metrics import MAE
from etna.models import NaiveModel
from etna.models import SimpleExpSmoothingModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform


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
    pipeline=Pipeline(NaiveModel()),
):
    trial = MagicMock(relative_params=relative_params)
    _objective = Tune.objective(
        ts=example_tsds,
        pipeline=pipeline,
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


def test_simple_tune_run(example_tsds, optuna_storage, pipeline=Pipeline(NaiveModel(1), horizon=7)):
    tune = Tune(
        pipeline,
        MAE(),
        metric_aggregation="median",
        horizon=7,
        storage=optuna_storage,
    )
    tune.fit(ts=example_tsds, n_trials=2)

    assert len(tune._optuna.study.trials) == 2
    assert len(tune.summary()) == 2
    assert len(tune.top_k()) == 2
    assert len(tune.top_k(k=1)) == 1


def test_can_handle_uniform(example_tsds, optuna_storage):
    params = {"model.smoothing_level": UniformDistribution(0.1, 1)}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(SimpleExpSmoothingModel(), horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_loguniform(example_tsds, optuna_storage):
    params = {"model.smoothing_level": LogUniformDistribution(0.1, 1)}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(SimpleExpSmoothingModel(), horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_discrete_uniform(example_tsds, optuna_storage):
    params = {"model.smoothing_level": DiscreteUniformDistribution(0.1, 1, 0.1)}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(SimpleExpSmoothingModel(), horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_intuniform(example_tsds, optuna_storage):
    params = {"model.lag": IntUniformDistribution(1, 6)}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(NaiveModel(), horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_intloguniform(example_tsds, optuna_storage):
    params = {"model.lag": IntLogUniformDistribution(1, 6)}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(NaiveModel(), horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_categorical(example_tsds, optuna_storage):
    params = {"model.lag": CategoricalDistribution((1, 2, 3))}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(NaiveModel(), horizon=7)
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)


def test_catch_nonexistent_argument_pass(example_tsds, optuna_storage):
    params = {"model.lag": UniformDistribution(1, 5)}
    with pytest.raises(TypeError):
        with patch.object(Pipeline, "params_to_tune", return_value=params):
            pipeline = Pipeline(SimpleExpSmoothingModel(), horizon=7)
            tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
            tune.fit(ts=example_tsds, n_trials=2)


def test_can_handle_transforms(example_tsds, optuna_storage):
    params = {"transforms.0.value": IntUniformDistribution(0, 17), "transforms.1.value": IntUniformDistribution(0, 17)}
    with patch.object(Pipeline, "params_to_tune", return_value=params):
        pipeline = Pipeline(
            NaiveModel(),
            [AddConstTransform(in_column="target", value=8), AddConstTransform(in_column="target", value=4)],
            horizon=7,
        )
        tune = Tune(pipeline, MAE(), metric_aggregation="median", horizon=7, storage=optuna_storage)
        tune.fit(ts=example_tsds, n_trials=2)
