import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.models import NaiveModel
from etna.transforms.decomposition import DeseasonalityTransform
from etna.transforms.decomposition.deseasonal import _OneSegmentDeseasonalityTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


def add_seasonality(series: pd.Series, period: int, magnitude: float) -> pd.Series:
    """Add seasonality to given series."""
    new_series = series.copy()
    size = series.shape[0]
    indices = np.arange(size)
    new_series += np.sin(2 * np.pi * indices / period) * magnitude
    return new_series


def get_one_df(period: int, magnitude: float) -> pd.DataFrame:
    df = pd.DataFrame()
    df["timestamp"] = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
    df["target"] = 10
    df["target"] = add_seasonality(df["target"], period=period, magnitude=magnitude)
    return df


def ts_seasonal() -> TSDataset:
    df_1 = get_one_df(period=7, magnitude=1)
    df_1["segment"] = "segment_1"
    df_2 = get_one_df(period=7, magnitude=2)
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq="D")


@pytest.fixture
def df_seasonal_one_segment() -> pd.DataFrame:
    df = get_one_df(period=7, magnitude=1)
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def df_seasonal_starting_with_nans_one_segment(df_seasonal_one_segment) -> pd.DataFrame:
    result = df_seasonal_one_segment
    result.iloc[:2] = np.NaN
    return result


@pytest.fixture
def ts_seasonal() -> TSDataset:
    df_1 = get_one_df(period=7, magnitude=1)
    df_1["segment"] = "segment_1"
    df_2 = get_one_df(period=7, magnitude=2)
    df_2["segment"] = "segment_2"
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset(TSDataset.to_dataset(classic_df), freq="D")


@pytest.fixture
def ts_seasonal_starting_with_nans() -> TSDataset:
    df_1 = get_one_df(period=7, magnitude=1)
    df_1["segment"] = "segment_1"

    df_2 = get_one_df(period=7, magnitude=2)
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.loc[[df.index[0], df.index[1]], pd.IndexSlice["segment_1", "target"]] = None
    return TSDataset(df, freq="D")


@pytest.fixture
def ts_seasonal_nan_tails() -> TSDataset:
    df_1 = get_one_df(period=7, magnitude=1)
    df_1["segment"] = "segment_1"

    df_2 = get_one_df(period=7, magnitude=2)
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    df.loc[[df.index[-2], df.index[-1]], pd.IndexSlice["segment_1", "target"]] = None
    return TSDataset(df, freq="D")


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("df_name", ["df_seasonal_one_segment", "df_seasonal_starting_with_nans_one_segment"])
def test_transform_one_segment(df_name, model, request):
    """Test that transform for one segment removes seasonality."""
    df = request.getfixturevalue(df_name)
    transform = _OneSegmentDeseasonalityTransform(in_column="target", period=7, model=model)
    df_transformed = transform.fit_transform(df)
    df_expected = df.copy()
    df_expected.loc[~df_expected["target"].isna(), "target"] = 10
    np.testing.assert_allclose(df_transformed["target"], df_expected["target"], atol=0.3)


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("ts_name", ["ts_seasonal", "ts_seasonal_starting_with_nans", "ts_seasonal_nan_tails"])
def test_transform_multi_segments(ts_name, model, request):
    """Test that transform for all segments removes seasonality."""
    ts = request.getfixturevalue(ts_name)
    df_expected = ts.to_pandas(flatten=True)
    df_expected.loc[~df_expected["target"].isna(), "target"] = 10
    transform = DeseasonalityTransform(in_column="target", period=7, model=model)
    transform.fit_transform(ts=ts)
    df_transformed = ts.to_pandas(flatten=True)
    np.testing.assert_allclose(df_transformed["target"], df_expected["target"], atol=0.3)


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("df_name", ["df_seasonal_one_segment", "df_seasonal_starting_with_nans_one_segment"])
def test_inverse_transform_one_segment(df_name, model, request):
    """Test that transform + inverse_transform don't change dataframe."""
    df = request.getfixturevalue(df_name)
    transform = _OneSegmentDeseasonalityTransform(in_column="target", period=7, model=model)
    df_transformed = transform.fit_transform(df)
    df_inverse_transformed = transform.inverse_transform(df_transformed)
    pd.util.testing.assert_frame_equal(df_inverse_transformed, df)


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
@pytest.mark.parametrize("ts_name", ["ts_seasonal", "ts_seasonal_starting_with_nans", "ts_seasonal_nan_tails"])
def test_inverse_transform_multi_segments(ts_name, model, request):
    """Test that transform + inverse_transform don't change tsdataset."""
    ts = request.getfixturevalue(ts_name)
    transform = DeseasonalityTransform(in_column="target", period=7, model=model)
    df = ts.to_pandas(flatten=True)
    transform.fit_transform(ts)
    transform.inverse_transform(ts)
    df_inverse_transformed = ts.to_pandas(flatten=True)
    pd.util.testing.assert_frame_equal(df_inverse_transformed, df)


@pytest.mark.parametrize("model_decompose", ["additive", "multiplicative"])
def test_forecast(ts_seasonal, model_decompose):
    """Test that transform works correctly in forecast."""
    transform = DeseasonalityTransform(in_column="target", period=7, model=model_decompose)
    ts_train, ts_test = ts_seasonal.train_test_split(test_size=3)
    transform.fit_transform(ts_train)
    model = NaiveModel()
    model.fit(ts_train)
    ts_future = ts_train.make_future(future_steps=3, transforms=[transform], tail_steps=model.context_size)
    ts_forecast = model.forecast(ts_future, prediction_size=3)
    ts_forecast.inverse_transform([transform])
    for segment in ts_forecast.segments:
        np.testing.assert_allclose(ts_forecast[:, segment, "target"], ts_test[:, segment, "target"], atol=0.1)


def test_transform_raise_error_if_not_fitted(df_seasonal_one_segment):
    """Test that transform for one segment raise error when calling transform without being fit."""
    transform = _OneSegmentDeseasonalityTransform(in_column="target", period=7, model="additive")
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=df_seasonal_one_segment)


def test_inverse_transform_raise_error_if_not_fitted(df_seasonal_one_segment):
    """Test that transform for one segment raise error when calling inverse_transform without being fit."""
    transform = _OneSegmentDeseasonalityTransform(in_column="target", period=7, model="additive")
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.inverse_transform(df=df_seasonal_one_segment)


def test_fit_transform_with_nans_in_middle_raise_error(ts_with_nans):
    transform = DeseasonalityTransform(in_column="target", period=7)
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        _ = transform.fit_transform(ts_with_nans)


@pytest.mark.parametrize("model", ["additive", "multiplicative"])
def test_transform_with_negative_input_values(df_seasonal_one_segment, model):
    df = df_seasonal_one_segment
    transform = _OneSegmentDeseasonalityTransform(in_column="target", period=7, model=model)
    transform.fit(df)
    df.iloc[:2] = -10
    if model == "additive":
        _ = transform.transform(df)
    else:
        with pytest.raises(
            ValueError,
            match="The input column contains zero or negative values,"
            "but multiplicative seasonality can not work with such values.",
        ):
            _ = transform.transform(df)


@pytest.mark.parametrize("model", ["mult"])
def test_not_allowed_model_name(model):
    with pytest.raises(
        NotImplementedError,
        match="mult is not a valid DeseasonalModel. Only 'additive', 'multiplicative' types allowed.",
    ):
        _ = DeseasonalityTransform(in_column="target", period=7, model=model)


@pytest.mark.parametrize(
    "transform",
    [
        DeseasonalityTransform(in_column="target", period=7, model="additive"),
        DeseasonalityTransform(in_column="target", period=7, model="multiplicative"),
    ],
)
def test_save_load(transform, ts_seasonal):
    assert_transformation_equals_loaded_original(transform=transform, ts=ts_seasonal)


def test_params_to_tune(ts_seasonal):
    ts = ts_seasonal
    transform = DeseasonalityTransform(in_column="target", period=7)
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts)
