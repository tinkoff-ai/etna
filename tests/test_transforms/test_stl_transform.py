import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.models import NaiveModel
from etna.transforms.stl import STLTransform
from etna.transforms.stl import _OneSegmentSTLTransform


def add_trend(series: pd.Series, coef: float = 1) -> pd.Series:
    """Add trend to given series."""
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += indices * coef
    return new_series


def add_seasonality(series: pd.Series, period: int, magnitude: float) -> pd.Series:
    """Add seasonality to given series."""
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += np.sin(2 * np.pi * indices / period) * magnitude
    return new_series


def get_one_df(coef: float, period: int, magnitude: float) -> pd.DataFrame:
    df = pd.DataFrame()
    df["timestamp"] = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
    df["target"] = 0
    df["target"] = add_seasonality(df["target"], period=period, magnitude=magnitude)
    df["target"] = add_trend(df["target"], coef=coef)
    return df


@pytest.fixture
def df_trend_seasonal_one_segment() -> pd.DataFrame:
    df = get_one_df(coef=0.1, period=7, magnitude=1)
    df.set_index("timestamp")
    return df


@pytest.fixture
def ts_trend_seasonal() -> TSDataset:
    df_1 = get_one_df(coef=0.1, period=7, magnitude=1)
    df_1["segment"] = "segment_1"
    df_2 = get_one_df(coef=0.05, period=7, magnitude=2)
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq="D")


@pytest.mark.parametrize("model", ["arima", "holt"])
def test_transform_one_segment(df_trend_seasonal_one_segment, model):
    """Test that transform for one segment removes trend and seasonality."""
    transform = _OneSegmentSTLTransform(in_column="target", period=7, model=model)
    df_transformed = transform.fit_transform(df_trend_seasonal_one_segment)
    np.testing.assert_allclose(df_transformed["target"], 0, atol=0.2)


@pytest.mark.parametrize("model", ["arima", "holt"])
def test_transform_multi_segments(ts_trend_seasonal, model):
    """Test that transform for all segments removes trend and seasonality."""
    transform = STLTransform(in_column="target", period=7, model=model)
    ts_trend_seasonal.fit_transform(transforms=[transform])
    np.testing.assert_allclose(ts_trend_seasonal[:, :, "target"], 0, atol=0.2)


@pytest.mark.parametrize("model", ["arima", "holt"])
def test_inverse_transform_one_segment(df_trend_seasonal_one_segment, model):
    """Test that transform + inverse_transform don't change dataframe."""
    transform = _OneSegmentSTLTransform(in_column="target", period=7, model=model)
    df_transformed = transform.fit_transform(df_trend_seasonal_one_segment)
    df_inverse_transformed = transform.inverse_transform(df_transformed)
    assert np.all(df_trend_seasonal_one_segment["target"] == df_inverse_transformed["target"])


@pytest.mark.parametrize("model", ["arima", "holt"])
def test_inverse_transform_multi_segments(ts_trend_seasonal, model):
    """Test that transform + inverse_transform don't change tsdataset."""
    transform = STLTransform(in_column="target", period=7, model=model)
    dataframe_initial = ts_trend_seasonal.to_pandas()
    ts_trend_seasonal.fit_transform(transforms=[transform])
    ts_trend_seasonal.inverse_transform()
    for segment in ts_trend_seasonal.segments:
        assert np.all(
            ts_trend_seasonal[:, segment, "target"] == dataframe_initial.loc[:, pd.IndexSlice[segment, "target"]]
        )


@pytest.mark.parametrize("model_stl", ["arima", "holt"])
def test_forecast(ts_trend_seasonal, model_stl):
    """Test that transform works correctly in forecast."""
    transform = STLTransform(in_column="target", period=7, model=model_stl)
    ts_train, ts_test = ts_trend_seasonal.train_test_split(
        ts_trend_seasonal.index[0],
        ts_trend_seasonal.index[-4],
        ts_trend_seasonal.index[-3],
        ts_trend_seasonal.index[-1],
    )
    ts_train.fit_transform(transforms=[transform])
    model = NaiveModel()
    model.fit(ts_train)
    ts_future = ts_train.make_future(3)
    ts_forecast = model.forecast(ts_future)
    for segment in ts_forecast.segments:
        np.testing.assert_allclose(ts_forecast[:, segment, "target"], ts_test[:, segment, "target"], atol=0.1)


def test_transform_raise_error_if_not_fitted(df_trend_seasonal_one_segment):
    """Test that transform for one segment raise error when calling transform without being fit."""
    transform = _OneSegmentSTLTransform(in_column="target", period=7, model="arima")
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df_trend_seasonal_one_segment)


def test_inverse_transform_raise_error_if_not_fitted(df_trend_seasonal_one_segment):
    """Test that transform for one segment raise error when calling transform without being fit."""
    transform = _OneSegmentSTLTransform(in_column="target", period=7, model="arima")
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.inverse_transform(df_trend_seasonal_one_segment)
