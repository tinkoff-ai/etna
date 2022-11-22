from copy import deepcopy
from datetime import datetime
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode
from etna.metrics import Width
from etna.models import CatBoostMultiSegmentModel
from etna.models import LinearPerSegmentModel
from etna.models import MovingAverageModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.models import SeasonalMovingAverageModel
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline import FoldMask
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import FilterFeaturesTransform
from etna.transforms import LagTransform
from etna.transforms import LogTransform
from etna.transforms import TimeSeriesImputerTransform
from tests.utils import DummyMetric

DEFAULT_METRICS = [MAE(mode=MetricAggregationMode.per_segment)]


@pytest.fixture
def ts_with_feature():
    periods = 100
    df = generate_ar_df(
        start_time="2019-01-01", periods=periods, ar_coef=[1], sigma=1, n_segments=2, random_seed=0, freq="D"
    )
    df_feature = generate_ar_df(
        start_time="2019-01-01", periods=periods, ar_coef=[0.9], sigma=2, n_segments=2, random_seed=42, freq="D"
    )
    df["feature_1"] = df_feature["target"].apply(lambda x: abs(x))
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.parametrize("horizon", ([1]))
def test_init_pass(horizon):
    """Check that Pipeline initialization works correctly in case of valid parameters."""
    pipeline = Pipeline(model=LinearPerSegmentModel(), transforms=[], horizon=horizon)
    assert pipeline.horizon == horizon


@pytest.mark.parametrize("horizon", ([-1]))
def test_init_fail(horizon):
    """Check that Pipeline initialization works correctly in case of invalid parameters."""
    with pytest.raises(ValueError, match="At least one point in the future is expected"):
        _ = Pipeline(
            model=LinearPerSegmentModel(),
            transforms=[],
            horizon=horizon,
        )


def test_fit(example_tsds):
    """Test that Pipeline correctly transforms dataset on fit stage."""
    original_ts = deepcopy(example_tsds)
    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    original_ts.fit_transform(transforms)
    original_ts.inverse_transform()
    assert np.all(original_ts.df.values == pipeline.ts.df.values)


@patch("etna.pipeline.pipeline.Pipeline._forecast")
def test_forecast_without_intervals_calls_private_forecast(private_forecast, example_tsds):
    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    _ = pipeline.forecast()

    private_forecast.assert_called()


@pytest.mark.parametrize(
    "model_class", [NonPredictionIntervalContextIgnorantAbstractModel, PredictionIntervalContextIgnorantAbstractModel]
)
def test_private_forecast_context_ignorant_model(model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline._forecast()

    ts.make_future.assert_called_with(future_steps=pipeline.horizon)
    model.forecast.assert_called_with(ts=ts.make_future())


@pytest.mark.parametrize(
    "model_class", [NonPredictionIntervalContextRequiredAbstractModel, PredictionIntervalContextRequiredAbstractModel]
)
def test_private_forecast_context_required_model(model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline._forecast()

    ts.make_future.assert_called_with(future_steps=pipeline.horizon, tail_steps=model.context_size)
    model.forecast.assert_called_with(ts=ts.make_future(), prediction_size=pipeline.horizon)


def test_forecast_with_intervals_prediction_interval_context_ignorant_model():
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=PredictionIntervalContextIgnorantAbstractModel)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))

    ts.make_future.assert_called_with(future_steps=pipeline.horizon)
    model.forecast.assert_called_with(ts=ts.make_future(), prediction_interval=True, quantiles=(0.025, 0.975))


def test_forecast_with_intervals_prediction_interval_context_required_model():
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=PredictionIntervalContextRequiredAbstractModel)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))

    ts.make_future.assert_called_with(future_steps=pipeline.horizon, tail_steps=model.context_size)
    model.forecast.assert_called_with(
        ts=ts.make_future(), prediction_size=pipeline.horizon, prediction_interval=True, quantiles=(0.025, 0.975)
    )


@patch("etna.pipeline.base.BasePipeline.forecast")
@pytest.mark.parametrize(
    "model_class",
    [NonPredictionIntervalContextIgnorantAbstractModel, NonPredictionIntervalContextRequiredAbstractModel],
)
def test_forecast_with_intervals_other_model(base_forecast, model_class):
    ts = MagicMock(spec=TSDataset)
    model = MagicMock(spec=model_class)

    pipeline = Pipeline(model=model, horizon=5)
    pipeline.fit(ts)
    _ = pipeline.forecast(prediction_interval=True, quantiles=(0.025, 0.975))
    base_forecast.assert_called_with(prediction_interval=True, quantiles=(0.025, 0.975), n_folds=3)


def test_forecast(example_tsds):
    """Test that the forecast from the Pipeline is correct."""
    original_ts = deepcopy(example_tsds)

    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    original_ts.fit_transform(transforms)
    model.fit(original_ts)
    future = original_ts.make_future(5)
    forecast_manual = model.forecast(future)

    assert np.all(forecast_pipeline.df.values == forecast_manual.df.values)


@pytest.mark.parametrize(
    "quantiles,prediction_interval_cv,error_msg",
    (
        [
            ([0.05, 1.5], 2, "Quantile should be a number from"),
            ([0.025, 0.975], 0, "Folds number should be a positive number, 0 given"),
        ]
    ),
)
def test_forecast_prediction_interval_incorrect_parameters(
    example_tsds, catboost_pipeline, quantiles, prediction_interval_cv, error_msg
):
    catboost_pipeline.fit(ts=deepcopy(example_tsds))
    with pytest.raises(ValueError, match=error_msg):
        _ = catboost_pipeline.forecast(quantiles=quantiles, n_folds=prediction_interval_cv)


@pytest.mark.parametrize("model", (ProphetModel(), SARIMAXModel()))
def test_forecast_prediction_interval_builtin(example_tsds, model):
    """Test that forecast method uses built-in prediction intervals for the listed models."""
    np.random.seed(1234)
    pipeline = Pipeline(model=model, transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast(prediction_interval=True)

    np.random.seed(1234)
    model = model.fit(example_tsds)
    future = example_tsds.make_future(5)
    forecast_model = model.forecast(ts=future, prediction_interval=True)

    assert forecast_model.df.equals(forecast_pipeline.df)


@pytest.mark.parametrize("model", (MovingAverageModel(), LinearPerSegmentModel()))
def test_forecast_prediction_interval_interface(example_tsds, model):
    """Test the forecast interface for the models without built-in prediction intervals."""
    pipeline = Pipeline(model=model, transforms=[DateFlagsTransform()], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_forecast_prediction_interval(splited_piecewise_constant_ts):
    """Test that the prediction interval for piecewise-constant dataset is correct."""
    train, test = splited_piecewise_constant_ts
    pipeline = Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=5)
    pipeline.fit(train)
    forecast = pipeline.forecast(prediction_interval=True)
    assert np.allclose(forecast.df.values, test.df.values)


@pytest.mark.parametrize("quantiles_narrow,quantiles_wide", ([([0.2, 0.8], [0.025, 0.975])]))
def test_forecast_prediction_interval_size(example_tsds, quantiles_narrow, quantiles_wide):
    """Test that narrow quantile levels gives more narrow interval than wide quantile levels."""
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles_narrow)
    narrow_interval_length = (
        forecast[:, :, f"target_{quantiles_narrow[1]}"].values - forecast[:, :, f"target_{quantiles_narrow[0]}"].values
    )

    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(example_tsds)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=quantiles_wide)
    wide_interval_length = (
        forecast[:, :, f"target_{quantiles_wide[1]}"].values - forecast[:, :, f"target_{quantiles_wide[0]}"].values
    )

    assert (narrow_interval_length <= wide_interval_length).all()


def test_forecast_prediction_interval_noise(constant_ts, constant_noisy_ts):
    """Test that prediction interval for noisy dataset is wider then for the dataset without noise."""
    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(constant_ts)
    forecast = pipeline.forecast(prediction_interval=True, quantiles=[0.025, 0.975])
    constant_interval_length = forecast[:, :, "target_0.975"].values - forecast[:, :, "target_0.025"].values

    pipeline = Pipeline(model=MovingAverageModel(), transforms=[], horizon=5)
    pipeline.fit(constant_noisy_ts)
    forecast = pipeline.forecast(prediction_interval=True)
    noisy_interval_length = forecast[:, :, "target_0.975"].values - forecast[:, :, "target_0.025"].values

    assert (constant_interval_length <= noisy_interval_length).all()


@pytest.mark.parametrize("n_folds", (0, -1))
def test_invalid_n_folds(catboost_pipeline: Pipeline, n_folds: int, example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior in case of invalid n_folds."""
    with pytest.raises(ValueError):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS, n_folds=n_folds)


def test_validate_backtest_dataset(catboost_pipeline_big: Pipeline, imbalanced_tsdf: TSDataset):
    """Test Pipeline.backtest behavior in case of small dataframe that
    can't be divided to required number of splits.
    """
    with pytest.raises(ValueError):
        _ = catboost_pipeline_big.backtest(ts=imbalanced_tsdf, n_folds=3, metrics=DEFAULT_METRICS)


@pytest.mark.parametrize("metrics", ([], [MAE(mode=MetricAggregationMode.macro)]))
def test_invalid_backtest_metrics(catboost_pipeline: Pipeline, metrics: List[Metric], example_tsdf: TSDataset):
    """Test Pipeline.backtest behavior in case of invalid metrics."""
    with pytest.raises(ValueError):
        _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=metrics, n_folds=2)


def test_generate_expandable_timeranges_days():
    """Test train-test timeranges generation in expand mode with daily freq"""
    df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", "2021-04-01")})
    df["segment"] = "seg"
    df["target"] = 1
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    ts = TSDataset(df, freq="D")

    true_borders = (
        (("2021-01-01", "2021-02-24"), ("2021-02-25", "2021-03-08")),
        (("2021-01-01", "2021-03-08"), ("2021-03-09", "2021-03-20")),
        (("2021-01-01", "2021-03-20"), ("2021-03-21", "2021-04-01")),
    )
    masks = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode="expand")
    for i, stage_dfs in enumerate(Pipeline._generate_folds_datasets(ts, masks=masks, horizon=12)):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d").date()


def test_generate_expandable_timeranges_hours():
    """Test train-test timeranges generation in expand mode with hour freq"""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-02-01", freq="H")})
    df["segment"] = "seg"
    df["target"] = 1
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    ts = TSDataset(df, freq="H")

    true_borders = (
        (("2020-01-01 00:00:00", "2020-01-30 12:00:00"), ("2020-01-30 13:00:00", "2020-01-31 00:00:00")),
        (("2020-01-01 00:00:00", "2020-01-31 00:00:00"), ("2020-01-31 01:00:00", "2020-01-31 12:00:00")),
        (("2020-01-01 00:00:00", "2020-01-31 12:00:00"), ("2020-01-31 13:00:00", "2020-02-01 00:00:00")),
    )
    masks = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode="expand")
    for i, stage_dfs in enumerate(Pipeline._generate_folds_datasets(ts, horizon=12, masks=masks)):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d %H:%M:%S").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d %H:%M:%S").date()


def test_generate_constant_timeranges_days():
    """Test train-test timeranges generation with constant mode with daily freq"""
    df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", "2021-04-01")})
    df["segment"] = "seg"
    df["target"] = 1
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    ts = TSDataset(df, freq="D")

    true_borders = (
        (("2021-01-01", "2021-02-24"), ("2021-02-25", "2021-03-08")),
        (("2021-01-13", "2021-03-08"), ("2021-03-09", "2021-03-20")),
        (("2021-01-25", "2021-03-20"), ("2021-03-21", "2021-04-01")),
    )
    masks = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode="constant")
    for i, stage_dfs in enumerate(Pipeline._generate_folds_datasets(ts, horizon=12, masks=masks)):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d").date()


def test_generate_constant_timeranges_hours():
    """Test train-test timeranges generation with constant mode with hours freq"""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-02-01", freq="H")})
    df["segment"] = "seg"
    df["target"] = 1
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    ts = TSDataset(df, freq="H")
    true_borders = (
        (("2020-01-01 00:00:00", "2020-01-30 12:00:00"), ("2020-01-30 13:00:00", "2020-01-31 00:00:00")),
        (("2020-01-01 12:00:00", "2020-01-31 00:00:00"), ("2020-01-31 01:00:00", "2020-01-31 12:00:00")),
        (("2020-01-02 00:00:00", "2020-01-31 12:00:00"), ("2020-01-31 13:00:00", "2020-02-01 00:00:00")),
    )
    masks = Pipeline._generate_masks_from_n_folds(ts=ts, n_folds=3, horizon=12, mode="constant")
    for i, stage_dfs in enumerate(Pipeline._generate_folds_datasets(ts, horizon=12, masks=masks)):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d %H:%M:%S").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d %H:%M:%S").date()


@pytest.mark.parametrize(
    "aggregate_metrics,expected_columns",
    (
        (
            False,
            ["fold_number", "MAE", "MSE", "segment", "SMAPE", DummyMetric("per-segment", alpha=0.0).__repr__()],
        ),
        (
            True,
            ["MAE", "MSE", "segment", "SMAPE", DummyMetric("per-segment", alpha=0.0).__repr__()],
        ),
    ),
)
def test_get_metrics_interface(
    catboost_pipeline: Pipeline, aggregate_metrics: bool, expected_columns: List[str], big_daily_example_tsdf: TSDataset
):
    """Check that Pipeline.backtest returns metrics in correct format."""
    metrics_df, _, _ = catboost_pipeline.backtest(
        ts=big_daily_example_tsdf,
        aggregate_metrics=aggregate_metrics,
        metrics=[MAE("per-segment"), MSE("per-segment"), SMAPE("per-segment"), DummyMetric("per-segment", alpha=0.0)],
    )
    assert sorted(expected_columns) == sorted(metrics_df.columns)


def test_get_forecasts_interface_daily(catboost_pipeline: Pipeline, big_daily_example_tsdf: TSDataset):
    """Check that Pipeline.backtest returns forecasts in correct format."""
    _, forecast, _ = catboost_pipeline.backtest(ts=big_daily_example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = sorted(
        ["regressor_lag_feature_10", "regressor_lag_feature_11", "regressor_lag_feature_12", "fold_number", "target"]
    )
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


def test_get_forecasts_interface_hours(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Check that Pipeline.backtest returns forecasts in correct format with non-daily seasonality."""
    _, forecast, _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = sorted(
        ["regressor_lag_feature_10", "regressor_lag_feature_11", "regressor_lag_feature_12", "fold_number", "target"]
    )
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


def test_get_fold_info_interface_daily(catboost_pipeline: Pipeline, big_daily_example_tsdf: TSDataset):
    """Check that Pipeline.backtest returns info dataframe in correct format."""
    _, _, info_df = catboost_pipeline.backtest(ts=big_daily_example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == sorted(info_df.columns)


def test_get_fold_info_interface_hours(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    """Check that Pipeline.backtest returns info dataframe in correct format with non-daily seasonality."""
    _, _, info_df = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == sorted(info_df.columns)


@pytest.mark.long_1
def test_backtest_with_n_jobs(catboost_pipeline: Pipeline, big_example_tsdf: TSDataset):
    """Check that Pipeline.backtest gives the same results in case of single and multiple jobs modes."""
    ts1 = deepcopy(big_example_tsdf)
    ts2 = deepcopy(big_example_tsdf)
    pipeline_1 = deepcopy(catboost_pipeline)
    pipeline_2 = deepcopy(catboost_pipeline)
    _, forecast_1, _ = pipeline_1.backtest(ts=ts1, n_jobs=1, metrics=DEFAULT_METRICS)
    _, forecast_2, _ = pipeline_2.backtest(ts=ts2, n_jobs=3, metrics=DEFAULT_METRICS)
    assert (forecast_1 == forecast_2).all().all()


def test_backtest_forecasts_sanity(step_ts: TSDataset):
    """Check that Pipeline.backtest gives correct forecasts according to the simple case."""
    ts, expected_metrics_df, expected_forecast_df = step_ts
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    metrics_df, forecast_df, _ = pipeline.backtest(ts, metrics=[MAE()], n_folds=3)

    assert np.all(metrics_df.reset_index(drop=True) == expected_metrics_df)
    assert np.all(forecast_df == expected_forecast_df)


def test_forecast_raise_error_if_not_fitted():
    """Test that Pipeline raise error when calling forecast without being fit."""
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    with pytest.raises(ValueError, match="Pipeline is not fitted!"):
        _ = pipeline.forecast()


def test_forecast_pipeline_with_nan_at_the_end(df_with_nans_in_tails):
    """Test that Pipeline can forecast with datasets with nans at the end."""
    pipeline = Pipeline(model=NaiveModel(), transforms=[TimeSeriesImputerTransform(strategy="forward_fill")], horizon=5)
    pipeline.fit(TSDataset(df_with_nans_in_tails, freq="1H"))
    forecast = pipeline.forecast()
    assert len(forecast.df) == 5


@pytest.mark.parametrize(
    "n_folds, mode, expected_masks",
    (
        (
            2,
            "expand",
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-03",
                    target_timestamps=["2020-04-04", "2020-04-05", "2020-04-06"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
        (
            2,
            "constant",
            [
                FoldMask(
                    first_train_timestamp="2020-01-01",
                    last_train_timestamp="2020-04-03",
                    target_timestamps=["2020-04-04", "2020-04-05", "2020-04-06"],
                ),
                FoldMask(
                    first_train_timestamp="2020-01-04",
                    last_train_timestamp="2020-04-06",
                    target_timestamps=["2020-04-07", "2020-04-08", "2020-04-09"],
                ),
            ],
        ),
    ),
)
def test_generate_masks_from_n_folds(example_tsds: TSDataset, n_folds, mode, expected_masks):
    masks = Pipeline._generate_masks_from_n_folds(ts=example_tsds, n_folds=n_folds, horizon=3, mode=mode)
    for mask, expected_mask in zip(masks, expected_masks):
        assert mask.first_train_timestamp == expected_mask.first_train_timestamp
        assert mask.last_train_timestamp == expected_mask.last_train_timestamp
        assert mask.target_timestamps == expected_mask.target_timestamps


@pytest.mark.parametrize(
    "mask", (FoldMask("2020-01-01", "2020-01-02", ["2020-01-03"]), FoldMask("2020-01-03", "2020-01-05", ["2020-01-06"]))
)
@pytest.mark.parametrize(
    "ts_name", ["simple_ts", "simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_generate_folds_datasets(ts_name, mask, request):
    """Check _generate_folds_datasets for correct work."""
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(lag=7))
    mask = pipeline._prepare_fold_masks(ts=ts, masks=[mask], mode="constant")[0]
    train, test = list(pipeline._generate_folds_datasets(ts, [mask], 4))[0]
    assert train.index.min() == np.datetime64(mask.first_train_timestamp)
    assert train.index.max() == np.datetime64(mask.last_train_timestamp)
    assert test.index.min() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(1, "D")
    assert test.index.max() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(4, "D")


@pytest.mark.parametrize(
    "mask", (FoldMask(None, "2020-01-02", ["2020-01-03"]), FoldMask(None, "2020-01-05", ["2020-01-06"]))
)
@pytest.mark.parametrize(
    "ts_name", ["simple_ts", "simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_generate_folds_datasets_without_first_date(ts_name, mask, request):
    """Check _generate_folds_datasets for correct work without first date."""
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(lag=7))
    mask = pipeline._prepare_fold_masks(ts=ts, masks=[mask], mode="constant")[0]
    train, test = list(pipeline._generate_folds_datasets(ts, [mask], 4))[0]
    assert train.index.min() == np.datetime64(ts.index.min())
    assert train.index.max() == np.datetime64(mask.last_train_timestamp)
    assert test.index.min() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(1, "D")
    assert test.index.max() == np.datetime64(mask.last_train_timestamp) + np.timedelta64(4, "D")


@pytest.mark.parametrize(
    "mask,expected",
    (
        (FoldMask("2020-01-01", "2020-01-07", ["2020-01-10"]), {"segment_0": 0, "segment_1": 11}),
        (FoldMask("2020-01-01", "2020-01-07", ["2020-01-08", "2020-01-11"]), {"segment_0": 95.5, "segment_1": 5}),
    ),
)
def test_run_fold(ts_run_fold: TSDataset, mask: FoldMask, expected: Dict[str, List[float]]):
    train, test = ts_run_fold.train_test_split(
        train_start=mask.first_train_timestamp, train_end=mask.last_train_timestamp
    )

    pipeline = Pipeline(model=NaiveModel(lag=5), transforms=[], horizon=4)
    fold = pipeline._run_fold(train, test, 1, mask, [MAE()], forecast_params=dict())
    for seg in fold["metrics"]["MAE"].keys():
        assert fold["metrics"]["MAE"][seg] == expected[seg]


@pytest.mark.parametrize(
    "lag,expected", ((5, {"segment_0": 76.923077, "segment_1": 90.909091}), (6, {"segment_0": 100, "segment_1": 120}))
)
def test_backtest_one_point(simple_ts: TSDataset, lag: int, expected: Dict[str, List[float]]):
    mask = FoldMask(
        simple_ts.index.min(),
        simple_ts.index.min() + np.timedelta64(6, "D"),
        [simple_ts.index.min() + np.timedelta64(8, "D")],
    )
    pipeline = Pipeline(model=NaiveModel(lag=lag), transforms=[], horizon=2)
    metrics_df, _, _ = pipeline.backtest(ts=simple_ts, metrics=[SMAPE()], n_folds=[mask], aggregate_metrics=True)
    metrics = dict(metrics_df.values)
    for segment in expected.keys():
        assert segment in metrics.keys()
        np.testing.assert_array_almost_equal(expected[segment], metrics[segment])


@pytest.mark.parametrize(
    "lag,expected", ((4, {"segment_0": 0, "segment_1": 0}), (7, {"segment_0": 0, "segment_1": 0.5}))
)
def test_backtest_two_points(masked_ts: TSDataset, lag: int, expected: Dict[str, List[float]]):
    mask = FoldMask(
        masked_ts.index.min(),
        masked_ts.index.min() + np.timedelta64(6, "D"),
        [masked_ts.index.min() + np.timedelta64(9, "D"), masked_ts.index.min() + np.timedelta64(10, "D")],
    )
    pipeline = Pipeline(model=NaiveModel(lag=lag), transforms=[], horizon=4)
    metrics_df, _, _ = pipeline.backtest(ts=masked_ts, metrics=[MAE()], n_folds=[mask], aggregate_metrics=True)
    metrics = dict(metrics_df.values)
    for segment in expected.keys():
        assert segment in metrics.keys()
        np.testing.assert_array_almost_equal(expected[segment], metrics[segment])


def test_sanity_backtest_naive_with_intervals(weekly_period_ts):
    train_ts, _ = weekly_period_ts
    quantiles = (0.01, 0.99)
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    _, forecast_df, _ = pipeline.backtest(
        ts=train_ts,
        metrics=[MAE(), Width(quantiles=quantiles)],
        forecast_params={"quantiles": quantiles, "prediction_interval": True},
    )
    features = forecast_df.columns.get_level_values(1)
    assert f"target_{quantiles[0]}" in features
    assert f"target_{quantiles[1]}" in features


@pytest.mark.long_1
def test_backtest_pass_with_filter_transform(ts_with_feature):
    ts = ts_with_feature

    pipeline = Pipeline(
        model=ProphetModel(),
        transforms=[
            LogTransform(in_column="feature_1"),
            FilterFeaturesTransform(exclude=["feature_1"], return_features=True),
        ],
        horizon=10,
    )
    pipeline.backtest(ts=ts, metrics=[MAE()], aggregate_metrics=True)


@pytest.mark.parametrize(
    "ts_name", ["simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_backtest_nans_at_beginning(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = Pipeline(model=NaiveModel(), horizon=2)
    _ = pipeline.backtest(
        ts=ts,
        metrics=[MAE()],
        n_folds=2,
    )


@pytest.mark.parametrize(
    "ts_name", ["simple_ts_starting_with_nans_one_segment", "simple_ts_starting_with_nans_all_segments"]
)
def test_backtest_nans_at_beginning_with_mask(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    mask = FoldMask(
        ts.index.min(),
        ts.index.min() + np.timedelta64(5, "D"),
        [ts.index.min() + np.timedelta64(6, "D"), ts.index.min() + np.timedelta64(8, "D")],
    )
    pipeline = Pipeline(model=NaiveModel(), horizon=3)
    _ = pipeline.backtest(
        ts=ts,
        metrics=[MAE()],
        n_folds=[mask],
    )


def test_forecast_backtest_correct_ordering(step_ts: TSDataset):
    ts, _, expected_forecast_df = step_ts
    pipeline = Pipeline(model=NaiveModel(), horizon=5)
    _, forecast_df, _ = pipeline.backtest(ts=ts, metrics=[MAE()], n_folds=3)
    assert np.all(forecast_df.values == expected_forecast_df.values)


def test_pipeline_with_deepmodel(example_tsds):
    from etna.models.nn import RNNModel

    pipeline = Pipeline(
        model=RNNModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1)),
        transforms=[],
        horizon=2,
    )
    _ = pipeline.backtest(ts=example_tsds, metrics=[MAE()], n_folds=2, aggregate_metrics=True)


@pytest.mark.parametrize(
    "model, transforms",
    [
        (
            CatBoostMultiSegmentModel(iterations=100),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(7, 15)))],
        ),
        (
            LinearPerSegmentModel(),
            [DateFlagsTransform(), LagTransform(in_column="target", lags=list(range(7, 15)))],
        ),
        (SeasonalMovingAverageModel(window=2, seasonality=7), []),
        (SARIMAXModel(), []),
        (ProphetModel(), []),
    ],
)
def test_predict(model, transforms, example_tsds):
    ts = example_tsds
    pipeline = Pipeline(model=model, transforms=transforms, horizon=7)
    pipeline.fit(ts)

    start_idx = 50
    end_idx = 70
    start_timestamp = ts.index[start_idx]
    end_timestamp = ts.index[end_idx]
    num_points = end_idx - start_idx + 1

    # create a separate TSDataset with slice of original timestamps
    predict_ts = deepcopy(ts)
    predict_ts.df = predict_ts.df.iloc[5 : end_idx + 5]

    result_ts = pipeline.predict(ts=predict_ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    result_df = result_ts.to_pandas(flatten=True)

    assert not np.any(result_df["target"].isna())
    assert len(result_df) == len(example_tsds.segments) * num_points
