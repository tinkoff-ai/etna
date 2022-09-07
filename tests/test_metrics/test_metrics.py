import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import sign
from etna.metrics import smape
from etna.metrics.base import MetricAggregationMode
from etna.metrics.metrics import MAE
from etna.metrics.metrics import MAPE
from etna.metrics.metrics import MSE
from etna.metrics.metrics import MSLE
from etna.metrics.metrics import R2
from etna.metrics.metrics import SMAPE
from etna.metrics.metrics import MedAE
from etna.metrics.metrics import Sign
from tests.utils import DummyMetric
from tests.utils import create_dummy_functional_metric


@pytest.mark.parametrize(
    "metric_class, metric_class_repr, metric_params, param_repr",
    (
        (MAE, "MAE", {}, ""),
        (MSE, "MSE", {}, ""),
        (MedAE, "MedAE", {}, ""),
        (MSLE, "MSLE", {}, ""),
        (MAPE, "MAPE", {}, ""),
        (SMAPE, "SMAPE", {}, ""),
        (R2, "R2", {}, ""),
        (Sign, "Sign", {}, ""),
        (DummyMetric, "DummyMetric", {"alpha": 1.0}, "alpha = 1.0, "),
    ),
)
def test_repr(metric_class, metric_class_repr, metric_params, param_repr):
    """Check metrics __repr__ method"""
    metric_mode = "per-segment"
    kwargs = {**metric_params, "kwarg_1": "value_1", "kwarg_2": "value_2"}
    kwargs_repr = param_repr + "kwarg_1 = 'value_1', kwarg_2 = 'value_2'"
    metric = metric_class(mode=metric_mode, **kwargs)
    metric_repr = metric.__repr__()
    true_repr = f"{metric_class_repr}(mode = '{metric_mode}', {kwargs_repr}, )"
    assert metric_repr == true_repr


@pytest.mark.parametrize(
    "metric_class",
    (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign),
)
def test_name_class_name(metric_class):
    """Check metrics name property without changing its during inheritance"""
    metric_mode = "per-segment"
    metric = metric_class(mode=metric_mode)
    metric_name = metric.name
    true_name = metric_class.__name__
    assert metric_name == true_name


@pytest.mark.parametrize(
    "metric_class",
    (DummyMetric,),
)
def test_name_repr(metric_class):
    """Check metrics name property with changing its during inheritance to repr"""
    metric_mode = "per-segment"
    metric = metric_class(mode=metric_mode)
    metric_name = metric.name
    true_name = metric.__repr__()
    assert metric_name == true_name


@pytest.mark.parametrize("metric_class", (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign))
def test_metrics_macro(metric_class, train_test_dfs):
    """Check metrics interface in 'macro' mode"""
    forecast_df, true_df = train_test_dfs
    metric = metric_class(mode=MetricAggregationMode.macro)
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, float)


@pytest.mark.parametrize("metric_class", (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_metrics_per_segment(metric_class, train_test_dfs):
    """Check metrics interface in 'per-segment' mode"""
    forecast_df, true_df = train_test_dfs
    metric = metric_class(mode=MetricAggregationMode.per_segment)
    value = metric(y_true=true_df, y_pred=forecast_df)
    assert isinstance(value, dict)
    for segment in forecast_df.df.columns.get_level_values("segment").unique():
        assert segment in value


@pytest.mark.parametrize("metric_class", (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_metrics_invalid_aggregation(metric_class):
    """Check metrics behavior in case of invalid aggregation mode"""
    with pytest.raises(NotImplementedError):
        _ = metric_class(mode="a")


@pytest.mark.parametrize("metric_class", (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_invalid_timestamps(metric_class, two_dfs_with_different_timestamps):
    """Check metrics behavior in case of invalid timeranges"""
    forecast_df, true_df = two_dfs_with_different_timestamps
    metric = metric_class()
    with pytest.raises(ValueError):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize("metric_class", (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_invalid_segments(metric_class, two_dfs_with_different_segments_sets):
    """Check metrics behavior in case of invalid segments sets"""
    forecast_df, true_df = two_dfs_with_different_segments_sets
    metric = metric_class()
    with pytest.raises(ValueError):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize("metric_class", (MAE, MSE, MedAE, MSLE, MAPE, SMAPE, R2, Sign, DummyMetric))
def test_invalid_segments_target(metric_class, train_test_dfs):
    """Check metrics behavior in case of no target column in segment"""
    forecast_df, true_df = train_test_dfs
    forecast_df.df.drop(columns=[("segment_1", "target")], inplace=True)
    metric = metric_class()
    with pytest.raises(ValueError):
        _ = metric(y_true=true_df, y_pred=forecast_df)


@pytest.mark.parametrize(
    "metric_class, metric_fn",
    (
        (MAE, mae),
        (MSE, mse),
        (MedAE, medae),
        (MSLE, msle),
        (MAPE, mape),
        (SMAPE, smape),
        (R2, r2_score),
        (Sign, sign),
        (DummyMetric, create_dummy_functional_metric()),
    ),
)
def test_metrics_values(metric_class, metric_fn, train_test_dfs):
    """
    Check that all the segments' metrics values in per-segments mode are equal to the same
    metric for segments' series.
    """
    forecast_df, true_df = train_test_dfs
    metric = metric_class(mode="per-segment")
    metric_values = metric(y_pred=forecast_df, y_true=true_df)
    for segment, value in metric_values.items():
        true_metric_value = metric_fn(
            y_true=true_df.loc[:, pd.IndexSlice[segment, "target"]],
            y_pred=forecast_df.loc[:, pd.IndexSlice[segment, "target"]],
        )
        assert value == true_metric_value


@pytest.mark.parametrize(
    "metric, greater_is_better",
    (
        (MAE(), False),
        (MSE(), False),
        (MedAE(), False),
        (MSLE(), False),
        (MAPE(), False),
        (SMAPE(), False),
        (R2(), True),
        (Sign(), None),
        (DummyMetric(), False),
    ),
)
def test_metrics_greater_is_better(metric, greater_is_better):
    assert metric.greater_is_better == greater_is_better


def test_multiple_calls():
    """Check that metric works correctly in case of multiple call."""
    timerange = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=10, freq="1D")})
    timestamp_base = pd.concat((timerange, timerange), axis=0)

    test_df_1 = timestamp_base.copy()
    test_df_2 = timestamp_base.copy()

    test_df_1["segment"] = ["A"] * 10 + ["B"] * 10
    test_df_2["segment"] = ["C"] * 10 + ["B"] * 10

    forecast_df_1 = test_df_1.copy()
    forecast_df_2 = test_df_2.copy()

    test_df_1["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    forecast_df_1["target"] = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 0, 3, 4, 5, 6, 7, 8, 9]

    test_df_2["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    forecast_df_2["target"] = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    test_df_1 = test_df_1.pivot(index="timestamp", columns="segment")
    test_df_1 = test_df_1.reorder_levels([1, 0], axis=1)
    test_df_1 = test_df_1.sort_index(axis=1)
    test_df_1.columns.names = ["segment", "feature"]

    test_df_2 = test_df_2.pivot(index="timestamp", columns="segment")
    test_df_2 = test_df_2.reorder_levels([1, 0], axis=1)
    test_df_2 = test_df_2.sort_index(axis=1)
    test_df_2.columns.names = ["segment", "feature"]

    forecast_df_1 = forecast_df_1.pivot(index="timestamp", columns="segment")
    forecast_df_1 = forecast_df_1.reorder_levels([1, 0], axis=1)
    forecast_df_1 = forecast_df_1.sort_index(axis=1)
    forecast_df_1.columns.names = ["segment", "feature"]

    forecast_df_2 = forecast_df_2.pivot(index="timestamp", columns="segment")
    forecast_df_2 = forecast_df_2.reorder_levels([1, 0], axis=1)
    forecast_df_2 = forecast_df_2.sort_index(axis=1)
    forecast_df_2.columns.names = ["segment", "feature"]

    test_df_1 = TSDataset(test_df_1, freq="1D")
    test_df_2 = TSDataset(test_df_2, freq="1D")
    forecast_df_1 = TSDataset(forecast_df_1, freq="1D")
    forecast_df_2 = TSDataset(forecast_df_2, freq="1D")

    metric = MAE(mode="per-segment")
    metric_value_1 = metric(y_true=test_df_1, y_pred=forecast_df_1)

    assert sorted(metric_value_1.keys()) == ["A", "B"]
    assert metric_value_1["A"] == 0.1
    assert metric_value_1["B"] == 0.2

    metric_value_2 = metric(y_true=test_df_2, y_pred=forecast_df_2)

    assert sorted(metric_value_2.keys()) == ["B", "C"]
    assert metric_value_2["C"] == 1
    assert metric_value_2["B"] == 0
