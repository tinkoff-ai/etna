from copy import deepcopy
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from etna.datasets.tsdataset import TSDataset
from etna.loggers import ConsoleLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics.base import MetricAggregationMode
from etna.model_selection.backtest import CrossValidationMode
from etna.model_selection.backtest import TimeSeriesCrossValidation
from etna.models.base import Model
from etna.models.catboost import CatBoostModelMultiSegment
from etna.models.linear import LinearPerSegmentModel
from etna.models.prophet import ProphetModel
from etna.transforms import DateFlagsTransform
from etna.transforms.base import Transform

DEFAULT_METRICS = [MAE(mode=MetricAggregationMode.per_segment)]


@pytest.fixture
def imbalanced_tsdf() -> TSDataset:
    """Generate two series with big time range difference"""
    df1 = pd.DataFrame({"timestamp": pd.date_range("2021-01-25", "2021-02-01", freq="D")})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(0, 5, len(df1))

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-02-01", freq="D")})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(0, 5, len(df2))

    df = df1.append(df2)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="1D")
    return df


@pytest.fixture()
def big_daily_example_tsdf() -> TSDataset:
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2019-01-01", end="2020-04-01", freq="D")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2019-06-01", end="2020-04-01", freq="D")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="1D")
    return df


@pytest.fixture()
def example_tsdf() -> TSDataset:
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="1H")
    return df


@pytest.fixture()
def big_example_tsdf() -> TSDataset:
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2020-01-01", end="2021-02-01", freq="D")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2020-01-01", end="2021-02-01", freq="D")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="1D")
    return df


def test_repr():
    """Check __repr__ method of TimeSeriesCrossValidation."""
    model = LinearPerSegmentModel(fit_intercept=True, normalize=False)
    mode = CrossValidationMode.expand.value
    tscv = TimeSeriesCrossValidation(model=model, horizon=12, n_folds=3, metrics=DEFAULT_METRICS, mode=mode)
    model_repr = model.__repr__()
    metrics_repr_inner = ", ".join([metric.__repr__() for metric in DEFAULT_METRICS])
    metrics_repr = f"[{metrics_repr_inner}]"
    mode_repr = CrossValidationMode[mode].__repr__()
    tscv_repr = tscv.__repr__()
    true_repr = (
        f"TimeSeriesCrossValidation(model = {model_repr}, horizon = 12, metrics = {metrics_repr}, "
        f"n_folds = 3, mode = {mode_repr}, n_jobs = 1, )"
    )
    assert tscv_repr == true_repr


@pytest.mark.parametrize("n_folds", (0, -1))
def test_invalid_n_split(n_folds: int):
    """Check TimeSeriesCrossValidation behavior in case of invalid n_folds"""
    with pytest.raises(ValueError):
        _ = TimeSeriesCrossValidation(model=ProphetModel(), horizon=12, metrics=DEFAULT_METRICS, n_folds=n_folds)


def test_invalid_metrics():
    """Check TimeSeriesCrossValidation behavior in case of invalid metrics"""
    with pytest.raises(ValueError):
        _ = TimeSeriesCrossValidation(
            model=CatBoostModelMultiSegment(), horizon=14, metrics=[MAE(mode=MetricAggregationMode.macro)]
        )


def test_validate_features(imbalanced_tsdf: TSDataset):
    """
    Check TimeSeriesCrossValidation behavior in case of small dataframe that
    can't be divided to required number of splits
    """
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tscv = TimeSeriesCrossValidation(model=CatBoostModelMultiSegment(), horizon=24, n_folds=3, metrics=DEFAULT_METRICS)
    with pytest.raises(ValueError):
        tscv.backtest(ts=imbalanced_tsdf, transforms=[date_flags])


def test_generate_expandable_timeranges_days():
    """Test train-test timeranges generation in expand mode with daily freq"""
    df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", "2021-04-01")})
    df["segment"] = "seg"
    df["target"] = 1
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="1D")

    true_borders = (
        (("2021-01-01", "2021-02-24"), ("2021-02-25", "2021-03-08")),
        (("2021-01-01", "2021-03-08"), ("2021-03-09", "2021-03-20")),
        (("2021-01-01", "2021-03-20"), ("2021-03-21", "2021-04-01")),
    )
    tscv = TimeSeriesCrossValidation(
        model=ProphetModel(), horizon=12, n_folds=3, metrics=DEFAULT_METRICS, mode=CrossValidationMode.expand.value
    )
    for i, stage_dfs in enumerate(tscv._generate_folds_dataframes(df)):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d").date()


def test_generate_expandable_timerange_hours():
    """Test train-test timeranges generation in expand mode with hour freq"""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-02-01", freq="H")})
    df["segment"] = "seg"
    df["target"] = 1
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="1H")

    true_borders = (
        (("2020-01-01 00:00:00", "2020-01-30 12:00:00"), ("2020-01-30 13:00:00", "2020-01-31 00:00:00")),
        (("2020-01-01 00:00:00", "2020-01-31 00:00:00"), ("2020-01-31 01:00:00", "2020-01-31 12:00:00")),
        (("2020-01-01 00:00:00", "2020-01-31 12:00:00"), ("2020-01-31 13:00:00", "2020-02-01 00:00:00")),
    )
    tscv = TimeSeriesCrossValidation(
        model=ProphetModel(), horizon=12, n_folds=3, metrics=DEFAULT_METRICS, mode=CrossValidationMode.expand.value
    )
    for i, stage_dfs in enumerate(tscv._generate_folds_dataframes(df)):
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
    df = TSDataset(df, freq="1D")

    true_borders = (
        (("2021-01-01", "2021-02-24"), ("2021-02-25", "2021-03-08")),
        (("2021-01-13", "2021-03-08"), ("2021-03-09", "2021-03-20")),
        (("2021-01-25", "2021-03-20"), ("2021-03-21", "2021-04-01")),
    )
    tscv = TimeSeriesCrossValidation(
        model=ProphetModel(), horizon=12, n_folds=3, metrics=DEFAULT_METRICS, mode=CrossValidationMode.constant.value
    )
    for i, stage_dfs in enumerate(tscv._generate_folds_dataframes(df)):
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
    df = TSDataset(df, freq="1H")
    true_borders = (
        (("2020-01-01 00:00:00", "2020-01-30 12:00:00"), ("2020-01-30 13:00:00", "2020-01-31 00:00:00")),
        (("2020-01-01 12:00:00", "2020-01-31 00:00:00"), ("2020-01-31 01:00:00", "2020-01-31 12:00:00")),
        (("2020-01-02 00:00:00", "2020-01-31 12:00:00"), ("2020-01-31 13:00:00", "2020-02-01 00:00:00")),
    )
    tscv = TimeSeriesCrossValidation(
        model=ProphetModel(), horizon=12, n_folds=3, metrics=DEFAULT_METRICS, mode=CrossValidationMode.constant.value
    )
    for i, stage_dfs in enumerate(tscv._generate_folds_dataframes(df)):
        for stage_df, borders in zip(stage_dfs, true_borders[i]):
            assert stage_df.index.min() == datetime.strptime(borders[0], "%Y-%m-%d %H:%M:%S").date()
            assert stage_df.index.max() == datetime.strptime(borders[1], "%Y-%m-%d %H:%M:%S").date()


def _fit_backtest_pipeline(
    model: Model, horizon: int, ts: TSDataset, transforms: Optional[List[Transform]] = [], n_jobs: int = 1
) -> TimeSeriesCrossValidation:
    """Init pipeline and run backtest"""
    tsvc = TimeSeriesCrossValidation(model=model, horizon=horizon, metrics=[MAE(), MSE(), SMAPE()], n_jobs=n_jobs)
    tsvc.backtest(ts=ts, transforms=transforms)
    return tsvc


@pytest.mark.parametrize(
    "aggregate_metrics,expected_columns",
    ((False, ["fold_number", "MAE", "MSE", "segment", "SMAPE"]), (True, ["MAE", "MSE", "segment", "SMAPE"])),
)
def test_get_metrics_interface(aggregate_metrics: bool, expected_columns: List[str], big_daily_example_tsdf: TSDataset):
    """Test interface of TimeSeriesCrossValidation.get_metrics with aggregate_metrics=False mode"""
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=big_daily_example_tsdf, transforms=[date_flags]
    )
    metrics_df = tsvc.get_metrics(aggregate_metrics=aggregate_metrics)
    assert sorted(expected_columns) == sorted(metrics_df.columns)


def test_get_forecasts_interface_daily(big_daily_example_tsdf: TSDataset):
    """Test interface of TimeSeriesCrossValidation.get_forecasts"""
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=big_daily_example_tsdf, transforms=[date_flags]
    )
    forecast = tsvc.get_forecasts()
    expected_columns = ["day_number_in_month", "day_number_in_week", "fold_number", "target"]
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


def test_get_forecasts_interface_hours(example_tsdf: TSDataset):
    """Test interface of TimeSeriesCrossValidation.get_forecasts"""
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=example_tsdf, transforms=[date_flags]
    )
    forecast = tsvc.get_forecasts()
    expected_columns = ["day_number_in_month", "day_number_in_week", "fold_number", "target"]
    assert expected_columns == sorted(set(forecast.columns.get_level_values("feature")))


def test_get_fold_info_interface_daily(big_daily_example_tsdf: TSDataset):
    """Test interface of TimeSeriesCrossValidation.get_fold_info"""
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=big_daily_example_tsdf, transforms=[date_flags]
    )
    forecast_df = tsvc.get_fold_info()
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == list(sorted(forecast_df.columns))


def test_get_fold_info_interface_hours(example_tsdf: TSDataset):
    """Test interface of TimeSeriesCrossValidation.get_fold_info"""
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=example_tsdf, transforms=[date_flags]
    )
    forecast_df = tsvc.get_fold_info()
    expected_columns = ["fold_number", "test_end_time", "test_start_time", "train_end_time", "train_start_time"]
    assert expected_columns == list(sorted(forecast_df.columns))


def test_logging(big_daily_example_tsdf: TSDataset):
    date_flags = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    file = NamedTemporaryFile()
    logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())
    metrics = [MAE(), MSE(), SMAPE()]
    metrics_str = ["MAE", "MSE", "SMAPE"]
    tsvc = TimeSeriesCrossValidation(model=CatBoostModelMultiSegment(), horizon=24, metrics=metrics, n_jobs=1)
    tsvc.backtest(ts=big_daily_example_tsdf, transforms=[date_flags])
    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == len(metrics) * tsvc.n_folds * len(big_daily_example_tsdf.segments)
        assert all([any([metric_str in line for metric_str in metrics_str]) for line in lines])
    tslogger.remove(idx)


@pytest.mark.long
def test_autoregressiveforecaster_backtest_pipeline(big_daily_example_tsdf: TSDataset):
    """This test checks that TimeSeriesCrossValidation works with AutoRegressiveForecaster"""
    tsvc = _fit_backtest_pipeline(model=ProphetModel(), horizon=12, ts=big_daily_example_tsdf)
    forecast = tsvc.get_forecasts()
    assert isinstance(forecast, pd.DataFrame)


@pytest.mark.long
def test_backtest_with_n_jobs(big_example_tsdf: TSDataset):
    """Check that backtest pipeline gives equal results in case of one and multiple jobs."""
    df1 = TSDataset(deepcopy(big_example_tsdf.df), freq=big_example_tsdf.freq)
    df2 = TSDataset(deepcopy(big_example_tsdf.df), freq=big_example_tsdf.freq)
    date_flags_1 = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    date_flags_2 = DateFlagsTransform(day_number_in_week=True, day_number_in_month=True)
    tsvc_1 = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=df1, transforms=[date_flags_1], n_jobs=1
    )
    tsvc_2 = _fit_backtest_pipeline(
        model=CatBoostModelMultiSegment(), horizon=24, ts=df2, transforms=[date_flags_2], n_jobs=3
    )
    forecast_1 = tsvc_1.get_forecasts()
    forecast_2 = tsvc_2.get_forecasts()
    assert (forecast_1 == forecast_2).all().all()
