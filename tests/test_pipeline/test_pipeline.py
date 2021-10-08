from copy import deepcopy
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import MetricAggregationMode
from etna.models import LinearPerSegmentModel
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.pipeline.backtest_utils import generate_folds_datasets
from etna.pipeline.backtest_utils import validate_backtest_dataset
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform

DEFAULT_METRICS = [MAE(mode=MetricAggregationMode.per_segment)]


def test_fit(example_tsds):
    """
    Test that Pipeline correctly transforms dataset on fit stage
    """
    original_ts = deepcopy(example_tsds)
    model = LinearPerSegmentModel()
    transforms = [AddConstTransform(in_column="target", value=10, inplace=True), DateFlagsTransform()]
    pipeline = Pipeline(model=model, transforms=transforms, horizon=5)
    pipeline.fit(example_tsds)
    original_ts.fit_transform(transforms)
    assert np.all(original_ts.df.values == pipeline.ts.df.values)


def test_forecast(example_tsds):
    """
    Test that the forecast from the Pipeline is correct
    """
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


def test_validate_backtest_dataset(imbalanced_tsdf: TSDataset):
    """Check validate_backtest_dataset behavior in case of small dataframe that
    can't be divided to required number of splits.
    """
    with pytest.raises(ValueError):
        validate_backtest_dataset(ts=imbalanced_tsdf, n_folds=3, horizon=24)


def test_invalid_metrics(example_tsds):
    """Check TimeSeriesCrossValidation behavior in case of invalid metrics"""
    pipeline = Pipeline(model=ProphetModel(), horizon=14)
    with pytest.raises(ValueError):
        metrics = [MAE(mode=MetricAggregationMode.macro)]
        pipeline.backtest(ts=example_tsds, metrics=metrics, n_folds=2)


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
    for i, stage_dfs in enumerate(generate_folds_datasets(ts, n_folds=3, horizon=12, mode="expand")):
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
    for i, stage_dfs in enumerate(generate_folds_datasets(ts, horizon=12, n_folds=3, mode="expand")):
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
    for i, stage_dfs in enumerate(generate_folds_datasets(ts, horizon=12, n_folds=3, mode="constant")):
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
    for i, stage_dfs in enumerate(generate_folds_datasets(ts, horizon=12, n_folds=3, mode="constant")):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d %H:%M:%S").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d %H:%M:%S").date()


@pytest.mark.parametrize(
    "aggregate_metrics,expected_columns",
    ((False, ["fold_number", "MAE", "MSE", "segment", "SMAPE"]), (True, ["MAE", "MSE", "segment", "SMAPE"])),
)
def test_get_metrics_interface(
    catboost_pipeline: Pipeline, aggregate_metrics: bool, expected_columns: List[str], big_daily_example_tsdf: TSDataset
):
    """Check that Pipeline.backtest returns metrics in correct format."""
    metrics_df, _, _ = catboost_pipeline.backtest(
        ts=big_daily_example_tsdf, aggregate_metrics=aggregate_metrics, metrics=[MAE(), MSE(), SMAPE()]
    )
    assert sorted(expected_columns) == sorted(metrics_df.columns)


def test_get_forecasts_interface_daily(catboost_pipeline: Pipeline, big_daily_example_tsdf: TSDataset):
    _, forecast, _ = catboost_pipeline.backtest(ts=big_daily_example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = sorted(
        ["regressor_target_lag_10", "regressor_target_lag_11", "regressor_target_lag_12", "fold_number", "target"]
    )
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


def test_get_forecasts_interface_hours(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    _, forecast, _ = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = sorted(
        ["regressor_target_lag_10", "regressor_target_lag_11", "regressor_target_lag_12", "fold_number", "target"]
    )
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


def test_get_fold_info_interface_daily(catboost_pipeline: Pipeline, big_daily_example_tsdf: TSDataset):
    _, _, info_df = catboost_pipeline.backtest(ts=big_daily_example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == list(sorted(info_df.columns))


def test_get_fold_info_interface_hours(catboost_pipeline: Pipeline, example_tsdf: TSDataset):
    _, _, info_df = catboost_pipeline.backtest(ts=example_tsdf, metrics=DEFAULT_METRICS)
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == list(sorted(info_df.columns))


@pytest.mark.long
def test_backtest_with_n_jobs(catboost_pipeline: Pipeline, big_example_tsdf: TSDataset):
    ts1 = deepcopy(big_example_tsdf)
    ts2 = deepcopy(big_example_tsdf)
    pipeline_1 = deepcopy(catboost_pipeline)
    pipeline_2 = deepcopy(catboost_pipeline)
    _, forecast_1, _ = pipeline_1.backtest(ts=ts1, n_jobs=1, metrics=DEFAULT_METRICS)
    _, forecast_2, _ = pipeline_2.backtest(ts=ts2, n_jobs=3, metrics=DEFAULT_METRICS)
    assert (forecast_1 == forecast_2).all().all()
