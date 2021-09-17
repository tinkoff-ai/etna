from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
from loguru import logger as _logger

from etna.datasets.tsdataset import TSDataset
from etna.loggers import ConsoleLogger
from etna.loggers import tslogger
from etna.models.linear import ElasticMultiSegmentModel
from etna.models.linear import ElasticPerSegmentModel
from etna.models.linear import LinearMultiSegmentModel
from etna.models.linear import LinearPerSegmentModel
from etna.transforms.lags import LagTransform


def linear_segments_by_parameters(alpha_values, intercept_values):
    dates = pd.date_range(start="2020-02-01", freq="D", periods=210)
    x = np.arange(210)
    train, test = [], []
    for i in range(3):
        train.append(pd.DataFrame())
        test.append(pd.DataFrame())
        train[i]["timestamp"], test[i]["timestamp"] = dates[:-7], dates[-7:]
        train[i]["segment"], test[i]["segment"] = f"segment_{i}", f"segment_{i}"

        alpha = alpha_values[i]
        intercept = intercept_values[i]
        target = x * alpha + intercept

        train[i]["target"], test[i]["target"] = target[:-7], target[-7:]

    train_df_all = pd.concat(train, ignore_index=True)
    test_df_all = pd.concat(test, ignore_index=True)
    train_ts = TSDataset(TSDataset.to_dataset(train_df_all), "D")
    test_ts = TSDataset(TSDataset.to_dataset(test_df_all), "D")

    return train_ts, test_ts


@pytest.fixture()
def linear_segments_ts_unique():
    """Create TSDataset that represents 3 segments with unique linear dependency on lags in each."""
    np.random.seed(42)
    alpha_values = [np.random.rand() * 4 - 2 for _ in range(3)]
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)


@pytest.fixture()
def linear_segments_ts_common():
    """Create TSDataset that represents 3 segments with common linear dependency on lags in each."""
    np.random.seed(42)
    alpha_values = [np.random.rand() * 4 - 2] * 3
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)


@pytest.mark.parametrize(
    "model_class, model_class_repr",
    ((LinearPerSegmentModel, "LinearPerSegmentModel"), (LinearMultiSegmentModel, "LinearMultiSegmentModel")),
)
def test_repr_linear(model_class, model_class_repr):
    """Check __repr__ method of LinearPerSegmentModel and LinearMultiSegmentModel."""
    kwargs = {"copy_X": True, "positive": True}
    kwargs_repr = "copy_X = True, positive = True"
    model = model_class(fit_intercept=True, normalize=False, **kwargs)
    model_repr = model.__repr__()
    true_repr = f"{model_class_repr}(fit_intercept = True, normalize = False, {kwargs_repr}, )"
    assert model_repr == true_repr


@pytest.mark.parametrize(
    "model_class, model_class_repr",
    ((ElasticPerSegmentModel, "ElasticPerSegmentModel"), (ElasticMultiSegmentModel, "ElasticMultiSegmentModel")),
)
def test_repr_elastic(model_class, model_class_repr):
    """Check __repr__ method of ElasticPerSegmentModel and ElasticMultiSegmentModel."""
    kwargs = {"copy_X": True, "positive": True}
    kwargs_repr = "copy_X = True, positive = True"
    model = model_class(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, **kwargs)
    model_repr = model.__repr__()
    true_repr = (
        f"{model_class_repr}(alpha = 1.0, l1_ratio = 0.5, " f"fit_intercept = True, normalize = False, {kwargs_repr}, )"
    )
    assert model_repr == true_repr


@pytest.mark.parametrize("model", [LinearPerSegmentModel(), ElasticPerSegmentModel()])
@pytest.mark.parametrize("num_lags", [3, 5, 10, 20, 30])
def test_model_per_segment(linear_segments_ts_unique, num_lags, model):
    """
    Given: Dataset with 3 linear segments and LinearRegression or ElasticNet model that predicts per segment
    When: Creating of lag features to target, applying it to dataset and making forecast for horizon periods
    Then: Predictions per segment is close to real values
    """
    horizon = 7
    train, test = linear_segments_ts_unique
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])

    model.fit(train)

    to_forecast = train.make_future(horizon)
    res = model.forecast(to_forecast)

    for segment in res.segments:
        assert np.allclose(test[:, segment, "target"], res[:, segment, "target"], atol=1)


@pytest.mark.parametrize("model", [LinearPerSegmentModel(), ElasticPerSegmentModel()])
def test_model_per_segment_logging(linear_segments_ts_unique, model):
    """Check working of logging in fit/forecast of per segment model."""
    horizon = 7
    train, test = linear_segments_ts_unique
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, 5 + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])

    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())

    model.fit(train)
    to_forecast = train.make_future(horizon)
    model.forecast(to_forecast)

    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == 2
        assert "fit" in lines[0]
        assert "forecast" in lines[1]

    tslogger.remove(idx)


@pytest.mark.parametrize("model", [LinearMultiSegmentModel(), ElasticMultiSegmentModel()])
@pytest.mark.parametrize("num_lags", [3, 5, 10, 20, 30])
def test_model_multi_segment(linear_segments_ts_common, num_lags, model):
    """
    Given: Dataset with 3 linear segments and LinearRegression or ElasticNet model that predicts across all segments
    When: Creating of lag features to target, applying it to dataset and making forecast for horizon periods
    Then: Predictions per segment is close to real values
    """
    horizon = 7
    train, test = linear_segments_ts_common
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, num_lags + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])

    model.fit(train)

    to_forecast = train.make_future(horizon)
    res = model.forecast(to_forecast)

    for segment in res.segments:
        assert np.allclose(test[:, segment, "target"], res[:, segment, "target"], atol=1)


@pytest.mark.parametrize("model", [LinearMultiSegmentModel(), ElasticMultiSegmentModel()])
def test_model_multi_segment_logging(linear_segments_ts_common, model):
    """Check working of logging in fit/forecast of multi segment model."""
    horizon = 7
    train, test = linear_segments_ts_common
    lags = LagTransform(in_column="target", lags=[i + horizon for i in range(1, 5 + 1)])
    train.fit_transform([lags])
    test.fit_transform([lags])

    file = NamedTemporaryFile()
    _logger.add(file.name)
    idx = tslogger.add(ConsoleLogger())

    model.fit(train)
    to_forecast = train.make_future(horizon)
    model.forecast(to_forecast)

    with open(file.name, "r") as in_file:
        lines = in_file.readlines()
        assert len(lines) == 2
        assert "fit" in lines[0]
        assert "forecast" in lines[1]

    tslogger.remove(idx)
