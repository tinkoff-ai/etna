from copy import deepcopy
from typing import List
from typing import Set
from typing import Union

import pandas as pd
import pytest
from joblib import Parallel
from joblib import delayed
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.metrics import MAE
from etna.pipeline import Pipeline

HORIZON = 7


def test_invalid_pipelines_number(catboost_pipeline: Pipeline):
    """Test StackingEnsemble behavior in case of invalid pipelines number."""
    with pytest.raises(ValueError):
        _ = StackingEnsemble(pipelines=[catboost_pipeline])


def test_get_horizon_pass(catboost_pipeline: Pipeline, prophet_pipeline: Pipeline):
    """Check that StackingEnsemble._get horizon works correctly in case of valid pipelines list."""
    horizon = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, prophet_pipeline])
    assert horizon == HORIZON


def test_get_horizon_fail(catboost_pipeline: Pipeline, naive_pipeline: Pipeline):
    """Check that StackingEnsemble._get horizon works correctly in case of invalid pipelines list."""
    with pytest.raises(ValueError):
        _ = StackingEnsemble._get_horizon(pipelines=[catboost_pipeline, naive_pipeline])


@pytest.mark.parametrize("input_cv,true_cv", ((2, 2), (None, 3)))
def test_cv_pass(input_cv, true_cv):
    """Check that StackingEnsemble._validate_cv works correctly in case of valid cv parameter."""
    cv = StackingEnsemble._validate_cv(input_cv)
    assert cv == true_cv


def test_cv_fail():
    """Check that StackingEnsemble._validate_cv works correctly in case of invalid cv parameter."""
    with pytest.raises(ValueError):
        _ = StackingEnsemble._validate_cv(cv=1)


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, None),
        (
            "all",
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_month",
                "regressor_day_number_in_week",
                "regressor_is_weekend",
            },
        ),
        (
            ["regressor_target_lag_10", "regressor_day_number_in_week", "unknown"],
            {"regressor_target_lag_10", "regressor_day_number_in_week"},
        ),
    ),
)
def test_features_to_use(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._validate_features_to_use works correctly."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(example_tsds))
        for pipeline in ensemble.pipelines
    )
    ensemble._validate_features_to_use(forecasts)
    features = ensemble.features_to_use
    assert features == expected_features


def test_stack_targets(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
):
    """Check that StackingEnsemble._stack_targets returns Dataframe with base models' forecasts in columns."""
    ensemble = StackingEnsemble(pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2])
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(example_tsds))
        for pipeline in ensemble.pipelines
    )
    stacked_targets = ensemble._stack_targets(forecasts=forecasts)
    columns = set(stacked_targets.columns.get_level_values("feature"))
    assert isinstance(stacked_targets, pd.DataFrame)
    assert len(stacked_targets) == len(forecasts[0].df)
    assert columns == {"regressor_target_0", "regressor_target_1"}


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (
            "all",
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_month",
                "regressor_day_number_in_week",
                "regressor_is_weekend",
            },
        ),
        (
            ["regressor_target_lag_10", "regressor_day_number_in_week", "unknown"],
            {"regressor_target_lag_10", "regressor_day_number_in_week"},
        ),
    ),
)
def test_get_features(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._get_features returns all the expected features in correct format."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(example_tsds))
        for pipeline in ensemble.pipelines
    )
    ensemble._validate_features_to_use(forecasts)
    features_df = ensemble._get_features(forecasts)
    features = set(features_df.columns.get_level_values("feature"))
    assert isinstance(features_df, pd.DataFrame)
    assert features == expected_features


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1", "target"}),
        (
            "all",
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_month",
                "regressor_day_number_in_week",
                "regressor_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
                "target",
            },
        ),
        (
            ["regressor_target_lag_10", "regressor_day_number_in_week", "unknown"],
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
                "target",
            },
        ),
    ),
)
def test_make_features_train(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._make_features works correctly in case of making features for train."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(example_tsds))
        for pipeline in ensemble.pipelines
    )
    ensemble._validate_features_to_use(forecasts)
    features_ts = ensemble._make_features(example_tsds, forecasts, train=True)
    features = set(features_ts.columns.get_level_values("feature"))
    targets_df = features_ts[:, :, "target"]
    assert isinstance(features_ts, TSDataset)
    assert features == expected_features
    assert (
        targets_df.values == example_tsds[targets_df.index.min() : targets_df.index.max(), :, "target"].values
    ).all()


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1", "target"}),
        (
            "all",
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_month",
                "regressor_day_number_in_week",
                "regressor_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
                "target",
            },
        ),
        (
            ["regressor_target_lag_10", "regressor_day_number_in_week", "unknown"],
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
                "target",
            },
        ),
    ),
)
def test_make_features_forecast(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._make_features works correctly in case of making features for forecast."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    forecasts = Parallel(n_jobs=ensemble.n_jobs, backend="multiprocessing", verbose=11)(
        delayed(ensemble._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(example_tsds))
        for pipeline in ensemble.pipelines
    )
    ensemble._validate_features_to_use(forecasts)
    features_ts = ensemble._make_features(example_tsds, forecasts, train=False)
    features = set(features_ts.columns.get_level_values("feature"))
    targets_df = features_ts[:, :, "target"]
    assert isinstance(features_ts, TSDataset)
    assert targets_df.isnull().all().all()
    assert features == expected_features


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_target_lag_10",
                "regressor_day_number_in_month",
                "regressor_day_number_in_week",
                "regressor_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_target_lag_10", "regressor_day_number_in_week", "unknown"],
            {"regressor_target_lag_10", "regressor_day_number_in_week", "regressor_target_0", "regressor_target_1"},
        ),
    ),
)
def test_forecast_interface(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble.forecast returns TSDataset of correct length, containing all the expected columns"""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    ensemble.fit(example_tsds)
    forecast = ensemble.forecast()
    features = set(forecast.columns.get_level_values("feature")) - {"target"}
    assert isinstance(forecast, TSDataset)
    assert len(forecast.df) == HORIZON
    assert features == expected_features


@pytest.mark.long
def test_multiprocessing_ensembles(
    simple_df: TSDataset,
    catboost_pipeline: Pipeline,
    prophet_pipeline: Pipeline,
    naive_pipeline_1: Pipeline,
    naive_pipeline_2: Pipeline,
):
    """Check that StackingEnsemble works the same in case of multi and single jobs modes."""
    pipelines = [catboost_pipeline, prophet_pipeline, naive_pipeline_1, naive_pipeline_2]
    single_jobs_ensemble = StackingEnsemble(pipelines=deepcopy(pipelines), n_jobs=1)
    multi_jobs_ensemble = StackingEnsemble(pipelines=deepcopy(pipelines), n_jobs=3)

    single_jobs_ensemble.fit(ts=deepcopy(simple_df))
    multi_jobs_ensemble.fit(ts=deepcopy(simple_df))

    single_jobs_forecast = single_jobs_ensemble.forecast()
    multi_jobs_forecast = multi_jobs_ensemble.forecast()

    assert (single_jobs_forecast.df == multi_jobs_forecast.df).all().all()


@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(stacking_ensemble_pipeline: StackingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that backtest works with StackingEnsemble."""
    results = stacking_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    for df in results:
        assert isinstance(df, pd.DataFrame)
