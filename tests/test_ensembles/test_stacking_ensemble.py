from copy import deepcopy
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.ensembles.stacking_ensemble import StackingEnsemble
from etna.metrics import MAE
from etna.pipeline import Pipeline

HORIZON = 7


@pytest.mark.parametrize("input_cv,true_cv", ([(2, 2)]))
def test_cv_pass(naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline, input_cv, true_cv):
    """Check that StackingEnsemble._validate_cv works correctly in case of valid cv parameter."""
    ensemble = StackingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], n_folds=input_cv)
    assert ensemble.n_folds == true_cv


@pytest.mark.parametrize("input_cv", ([0]))
def test_cv_fail_wrong_number(naive_pipeline_1: Pipeline, naive_pipeline_2: Pipeline, input_cv):
    """Check that StackingEnsemble._validate_cv works correctly in case of wrong number for cv parameter."""
    with pytest.raises(ValueError, match="Folds number should be a positive number, 0 given"):
        _ = StackingEnsemble(pipelines=[naive_pipeline_1, naive_pipeline_2], n_folds=input_cv)


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, None),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week"],
            {"regressor_lag_feature_10", "regressor_dateflag_day_number_in_week"},
        ),
    ),
)
def test_features_to_use(
    forecasts_ts: TSDataset,
    naive_featured_pipeline_1,
    naive_featured_pipeline_2,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._get_features_to_use works correctly."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    obtained_features = ensemble._filter_features_to_use(forecasts_ts)
    assert obtained_features == expected_features


@pytest.mark.parametrize("features_to_use", (["regressor_lag_feature_10"]))
def test_features_to_use_wrong_format(
    forecasts_ts: TSDataset,
    naive_featured_pipeline_1,
    naive_featured_pipeline_2,
    features_to_use: Union[None, Literal[all], List[str]],
):
    """Check that StackingEnsemble._get_features_to_use raises warning in case of wrong format."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    with pytest.warns(UserWarning, match="Feature list is passed in the wrong format."):
        _ = ensemble._filter_features_to_use(forecasts_ts)


@pytest.mark.parametrize("features_to_use", ([["unknown_feature"]]))
def test_features_to_use_not_found(
    forecasts_ts: TSDataset,
    naive_featured_pipeline_1,
    naive_featured_pipeline_2,
    features_to_use: Union[None, Literal[all], List[str]],
):
    """Check that StackingEnsemble._get_features_to_use raises warning in case of unavailable features."""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    )
    with pytest.warns(UserWarning, match=f"Features {set(features_to_use)} are not found and will be dropped!"):
        _ = ensemble._filter_features_to_use(forecasts_ts)


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week", "unknown"],
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
    ),
)
def test_make_features(
    example_tsds,
    forecasts_ts,
    targets,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble._make_features returns X,y with all the expected columns
    and which are compatible with the sklearn interface.
    """
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    ).fit(example_tsds)
    x, y = ensemble._make_features(forecasts_ts, train=True)
    features = set(x.columns.get_level_values("feature"))
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert features == expected_features
    assert (y == targets).all()


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week", "unknown"],
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
            },
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
    ).fit(example_tsds)
    forecast = ensemble.forecast()
    features = set(forecast.columns.get_level_values("feature")) - {"target"}
    assert isinstance(forecast, TSDataset)
    assert len(forecast.df) == HORIZON
    assert features == expected_features


@pytest.mark.parametrize(
    "features_to_use,expected_features",
    (
        (None, {"regressor_target_0", "regressor_target_1"}),
        (
            "all",
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_month",
                "regressor_dateflag_day_number_in_week",
                "regressor_dateflag_is_weekend",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
        (
            ["regressor_lag_feature_10", "regressor_dateflag_day_number_in_week", "unknown"],
            {
                "regressor_lag_feature_10",
                "regressor_dateflag_day_number_in_week",
                "regressor_target_0",
                "regressor_target_1",
            },
        ),
    ),
)
def test_predict_interface(
    example_tsds,
    naive_featured_pipeline_1: Pipeline,
    naive_featured_pipeline_2: Pipeline,
    features_to_use: Union[None, Literal[all], List[str]],
    expected_features: Set[str],
):
    """Check that StackingEnsemble.predict returns TSDataset of correct length, containing all the expected columns"""
    ensemble = StackingEnsemble(
        pipelines=[naive_featured_pipeline_1, naive_featured_pipeline_2], features_to_use=features_to_use
    ).fit(example_tsds)
    start_idx = 20
    end_idx = 30
    prediction = ensemble.predict(
        ts=example_tsds, start_timestamp=example_tsds.index[start_idx], end_timestamp=example_tsds.index[end_idx]
    )
    features = set(prediction.columns.get_level_values("feature")) - {"target"}
    assert isinstance(prediction, TSDataset)
    assert len(prediction.df) == end_idx - start_idx + 1
    assert features == expected_features


def test_forecast_prediction_interval_interface(example_tsds, naive_ensemble: StackingEnsemble):
    """Test the forecast interface with prediction intervals."""
    naive_ensemble.fit(example_tsds)
    forecast = naive_ensemble.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_forecast_calls_process_forecasts(example_tsds: TSDataset, naive_ensemble):
    naive_ensemble.fit(ts=example_tsds)
    naive_ensemble._process_forecasts = MagicMock()

    result = naive_ensemble._forecast()

    naive_ensemble._process_forecasts.assert_called_once()
    assert result == naive_ensemble._process_forecasts.return_value


def test_predict_calls_process_forecasts(example_tsds: TSDataset, naive_ensemble):
    naive_ensemble.fit(ts=example_tsds)
    naive_ensemble._process_forecasts = MagicMock()

    result = naive_ensemble._predict(
        ts=example_tsds,
        start_timestamp=example_tsds.index[20],
        end_timestamp=example_tsds.index[30],
        prediction_interval=False,
        quantiles=(),
    )

    naive_ensemble._process_forecasts.assert_called_once()
    assert result == naive_ensemble._process_forecasts.return_value


def test_forecast_sanity(weekly_period_ts: Tuple["TSDataset", "TSDataset"], naive_ensemble: StackingEnsemble):
    """Check that StackingEnsemble.forecast forecast correct values"""
    train, test = weekly_period_ts
    ensemble = naive_ensemble.fit(train)
    forecast = ensemble.forecast()
    mae = MAE("macro")
    np.allclose(mae(test, forecast), 0)


@pytest.mark.long_1
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


@pytest.mark.long_1
@pytest.mark.parametrize("n_jobs", (1, 5))
def test_backtest(stacking_ensemble_pipeline: StackingEnsemble, example_tsds: TSDataset, n_jobs: int):
    """Check that backtest works with StackingEnsemble."""
    results = stacking_ensemble_pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_jobs=n_jobs, n_folds=3)
    for df in results:
        assert isinstance(df, pd.DataFrame)


def test_forecast_raise_error_if_not_fitted(naive_ensemble: StackingEnsemble):
    """Test that StackingEnsemble raise error when calling forecast without being fit."""
    with pytest.raises(ValueError, match="StackingEnsemble is not fitted!"):
        _ = naive_ensemble.forecast()
