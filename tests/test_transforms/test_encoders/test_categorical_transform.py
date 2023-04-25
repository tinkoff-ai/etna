from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.datasets import generate_const_df
from etna.datasets import generate_periodic_df
from etna.metrics import R2
from etna.models import LinearPerSegmentModel
from etna.transforms import FilterFeaturesTransform
from etna.transforms.encoders.categorical import LabelEncoderTransform
from etna.transforms.encoders.categorical import OneHotEncoderTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


def get_two_ts_with_new_values(dtype: str = "int"):
    dct_1 = {
        "timestamp": list(pd.date_range(start="2021-01-01", end="2021-01-03")) * 2,
        "segment": ["segment_0"] * 3 + ["segment_1"] * 3,
        "regressor_0": [5, 8, 5, 9, 5, 9],
        "target": [1, 2, 3, 4, 5, 6],
    }
    df_1 = pd.DataFrame(dct_1)
    df_1["regressor_0"] = df_1["regressor_0"].astype(dtype)
    df_1 = TSDataset.to_dataset(df_1)
    ts_1 = TSDataset(df=df_1, freq="D")

    dct_2 = {
        "timestamp": list(pd.date_range(start="2021-01-01", end="2021-01-03")) * 2,
        "segment": ["segment_0"] * 3 + ["segment_1"] * 3,
        "regressor_0": [5, 8, 9, 5, 0, 0],
        "target": [1, 2, 3, 4, 5, 6],
    }
    df_2 = pd.DataFrame(dct_2)
    df_2["regressor_0"] = df_2["regressor_0"].astype(dtype)
    df_2 = TSDataset.to_dataset(df_2)
    ts_2 = TSDataset(df=df_2, freq="D")

    return ts_1, ts_2


@pytest.fixture
def two_ts_with_new_values():
    return get_two_ts_with_new_values()


def get_ts_for_ohe_encoding(dtype: str = "int"):
    df_to_forecast = generate_ar_df(10, start_time="2021-01-01", n_segments=1)
    d = {
        "timestamp": pd.date_range(start="2021-01-01", end="2021-01-12"),
        "regressor_0": [5, 8, 5, 8, 5, 8, 5, 8, 5, 8, 5, 8],
        "regressor_1": [9, 5, 9, 5, 9, 5, 9, 5, 9, 5, 9, 5],
        "regressor_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    df_regressors = pd.DataFrame(d)
    regressor_cols = ["regressor_0", "regressor_1", "regressor_2"]
    df_regressors[regressor_cols] = df_regressors[regressor_cols].astype(dtype)
    df_regressors["segment"] = "segment_0"

    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors, known_future=regressor_cols)

    answer_on_regressor_0 = tsdataset.df.copy()["segment_0"]
    answer_on_regressor_0["test_0"] = answer_on_regressor_0["regressor_0"].apply(lambda x: int(int(x) == 5))
    answer_on_regressor_0["test_1"] = answer_on_regressor_0["regressor_0"].apply(lambda x: int(int(x) == 8))
    answer_on_regressor_0["test_0"] = answer_on_regressor_0["test_0"].astype("category")
    answer_on_regressor_0["test_1"] = answer_on_regressor_0["test_1"].astype("category")

    answer_on_regressor_1 = tsdataset.df.copy()["segment_0"]
    answer_on_regressor_1["test_0"] = answer_on_regressor_1["regressor_1"].apply(lambda x: int(int(x) == 5))
    answer_on_regressor_1["test_1"] = answer_on_regressor_1["regressor_1"].apply(lambda x: int(int(x) == 9))
    answer_on_regressor_1["test_0"] = answer_on_regressor_1["test_0"].astype("category")
    answer_on_regressor_1["test_1"] = answer_on_regressor_1["test_1"].astype("category")

    answer_on_regressor_2 = tsdataset.df.copy()["segment_0"]
    answer_on_regressor_2["test_0"] = answer_on_regressor_2["regressor_2"].apply(lambda x: int(int(x) == 0))
    answer_on_regressor_2["test_0"] = answer_on_regressor_2["test_0"].astype("category")

    return tsdataset, (answer_on_regressor_0, answer_on_regressor_1, answer_on_regressor_2)


@pytest.fixture
def ts_for_ohe_encoding():
    return get_ts_for_ohe_encoding()


def get_ts_for_label_encoding(dtype: str = "int"):
    df_to_forecast = generate_ar_df(10, start_time="2021-01-01", n_segments=1)
    d = {
        "timestamp": pd.date_range(start="2021-01-01", end="2021-01-12"),
        "regressor_0": [5, 8, 5, 8, 5, 8, 5, 8, 5, 8, 5, 8],
        "regressor_1": [9, 5, 9, 5, 9, 5, 9, 5, 9, 5, 9, 5],
        "regressor_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    df_regressors = pd.DataFrame(d)
    regressor_cols = ["regressor_0", "regressor_1", "regressor_2"]
    df_regressors[regressor_cols] = df_regressors[regressor_cols].astype(dtype)
    df_regressors["segment"] = "segment_0"

    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors, known_future=regressor_cols)

    answer_on_regressor_0 = tsdataset.df.copy()["segment_0"]
    answer_on_regressor_0["test"] = answer_on_regressor_0["regressor_0"].apply(lambda x: float(int(x) == 8))
    answer_on_regressor_0["test"] = answer_on_regressor_0["test"].astype("category")

    answer_on_regressor_1 = tsdataset.df.copy()["segment_0"]
    answer_on_regressor_1["test"] = answer_on_regressor_1["regressor_1"].apply(lambda x: float(int(x) == 9))
    answer_on_regressor_1["test"] = answer_on_regressor_1["test"].astype("category")

    answer_on_regressor_2 = tsdataset.df.copy()["segment_0"]
    answer_on_regressor_2["test"] = answer_on_regressor_2["regressor_2"].apply(lambda x: float(int(x) == 1))
    answer_on_regressor_2["test"] = answer_on_regressor_2["test"].astype("category")

    return tsdataset, (answer_on_regressor_0, answer_on_regressor_1, answer_on_regressor_2)


@pytest.fixture
def ts_for_label_encoding():
    return get_ts_for_label_encoding()


@pytest.fixture
def ts_for_naming():
    df_to_forecast = generate_ar_df(10, start_time="2021-01-01", n_segments=1)
    df_regressors = generate_periodic_df(12, start_time="2021-01-01", scale=10, period=2, n_segments=2)
    df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    df_regressors.columns = ["timestamp"] + ["regressor_1", "2"]
    df_regressors["segment"] = "segment_0"
    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors, known_future=["regressor_1"])
    return tsdataset


@pytest.mark.parametrize("dtype", ["float", "int", "str", "category"])
def test_label_encoder_simple(dtype):
    """Test that LabelEncoderTransform works correct in a simple cases."""
    ts, answers = get_ts_for_label_encoding(dtype=dtype)
    for i in range(3):
        le = LabelEncoderTransform(in_column=f"regressor_{i}", out_column="test")
        le.fit(ts)
        cols = le._get_column_name()
        df_transformed = le.transform(deepcopy(ts)).to_pandas()["segment_0"][cols]
        df_expected = answers[i][cols]
        assert df_transformed.equals(df_expected)


@pytest.mark.parametrize("dtype", ["float", "int", "str", "category"])
def test_ohe_encoder_simple(dtype):
    """Test that OneHotEncoderTransform works correct in a simple case."""
    ts, answers = get_ts_for_ohe_encoding(dtype)
    for i in range(3):
        ohe = OneHotEncoderTransform(in_column=f"regressor_{i}", out_column="test")
        ohe.fit(ts)
        cols = ohe._get_out_column_names()
        df_transformed = ohe.transform(deepcopy(ts)).to_pandas()["segment_0"][cols]
        df_expected = answers[i][cols]
        assert df_transformed.equals(df_expected)


def test_value_error_label_encoder(ts_for_label_encoding):
    """Test LabelEncoderTransform with wrong strategy."""
    ts, _ = ts_for_label_encoding
    with pytest.raises(ValueError, match="The strategy"):
        le = LabelEncoderTransform(in_column="target", strategy="fake_strategy")
        le.fit(ts)
        le.transform(ts)


@pytest.mark.parametrize(
    "strategy, expected_values",
    [
        ("new_value", {"segment_0": [0, 1, 2], "segment_1": [0, -1, -1]}),
        ("none", {"segment_0": [0, 1, 2], "segment_1": [0, np.nan, np.nan]}),
        ("mean", {"segment_0": [0, 1, 2], "segment_1": [0, 3 / 4, 3 / 4]}),
    ],
)
@pytest.mark.parametrize("dtype", ["float", "int", "str", "category"])
def test_new_value_label_encoder(dtype, strategy, expected_values):
    """Test LabelEncoderTransform correct works with unknown values."""
    ts1, ts2 = get_two_ts_with_new_values(dtype=dtype)
    segments = ts1.segments
    le = LabelEncoderTransform(in_column="regressor_0", strategy=strategy, out_column="encoded_regressor_0")
    le.fit(ts1)
    df2_transformed = le.transform(ts2).to_pandas()
    for segment in segments:
        values = df2_transformed.loc[:, pd.IndexSlice[segment, "encoded_regressor_0"]].values
        np.testing.assert_array_almost_equal(values, expected_values[segment])


@pytest.mark.parametrize(
    "expected_values",
    [{"segment_0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "segment_1": [[1, 0, 0], [0, 0, 0], [0, 0, 0]]}],
)
@pytest.mark.parametrize("dtype", ["float", "int", "str", "category"])
def test_new_value_ohe_encoder(dtype, expected_values):
    """Test OneHotEncoderTransform correct works with unknown values."""
    ts1, ts2 = get_two_ts_with_new_values(dtype=dtype)
    segments = ts1.segments
    out_columns = ["targets_0", "targets_1", "targets_2"]
    ohe = OneHotEncoderTransform(in_column="regressor_0", out_column="targets")
    ohe.fit(ts1)
    df2_transformed = ohe.transform(ts2).to_pandas()
    for segment in segments:
        values = df2_transformed.loc[:, pd.IndexSlice[segment, out_columns]].values
        np.testing.assert_array_almost_equal(values, expected_values[segment])


def test_naming_ohe_encoder(two_ts_with_new_values):
    """Test OneHotEncoderTransform gives the correct columns."""
    ts1, ts2 = two_ts_with_new_values
    ohe = OneHotEncoderTransform(in_column="regressor_0", out_column="targets")
    ohe.fit(ts1)
    segments = ["segment_0", "segment_1"]
    target = ["target", "targets_0", "targets_1", "targets_2", "regressor_0"]
    assert {(i, j) for i in segments for j in target} == set(ohe.transform(ts2).columns.values)


@pytest.mark.parametrize(
    "in_column",
    [("2"), ("regressor_1")],
)
def test_naming_ohe_encoder_no_out_column(ts_for_naming, in_column):
    """Test OneHotEncoderTransform gives the correct columns with no out_column."""
    ts = ts_for_naming
    ohe = OneHotEncoderTransform(in_column=in_column)
    ohe.fit(ts)
    answer = set(list(ts.to_pandas()["segment_0"].columns) + [str(ohe.__repr__()) + "_0", str(ohe.__repr__()) + "_1"])
    assert answer == set(ohe.transform(ts).to_pandas()["segment_0"].columns.values)


@pytest.mark.parametrize(
    "in_column",
    [("2"), ("regressor_1")],
)
def test_naming_label_encoder_no_out_column(ts_for_naming, in_column):
    """Test LabelEncoderTransform gives the correct columns with no out_column."""
    ts = ts_for_naming
    le = LabelEncoderTransform(in_column=in_column)
    le.fit(ts)
    answer = set(list(ts.to_pandas()["segment_0"].columns) + [str(le.__repr__())])
    assert answer == set(le.transform(ts).to_pandas()["segment_0"].columns.values)


@pytest.fixture
def ts_for_ohe_sanity():
    df_to_forecast = generate_const_df(periods=100, start_time="2021-01-01", scale=0, n_segments=1)
    df_regressors = generate_periodic_df(periods=120, start_time="2021-01-01", scale=10, period=4, n_segments=1)
    df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    df_regressors.columns = ["timestamp"] + [f"regressor_{i}" for i in range(1)]
    df_regressors["segment"] = "segment_0"
    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    rng = np.random.default_rng(12345)

    def f(x):
        return x**2 + rng.normal(0, 0.01)

    df_to_forecast["segment_0", "target"] = df_regressors["segment_0"]["regressor_0"][:100].apply(f)
    ts = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors, known_future="all")
    return ts


def test_ohe_sanity(ts_for_ohe_sanity):
    """Test for correct work in the full forecasting pipeline."""
    horizon = 10
    train_ts, test_ts = ts_for_ohe_sanity.train_test_split(test_size=horizon)
    ohe = OneHotEncoderTransform(in_column="regressor_0")
    filt = FilterFeaturesTransform(exclude=["regressor_0"])
    transforms = [ohe, filt]
    train_ts.fit_transform(transforms)
    model = LinearPerSegmentModel()
    model.fit(train_ts)
    future_ts = train_ts.make_future(horizon, transforms=transforms)
    forecast_ts = model.forecast(future_ts)
    forecast_ts.inverse_transform(transforms)
    r2 = R2()
    assert 1 - r2(test_ts, forecast_ts)["segment_0"] < 1e-5


@pytest.mark.filterwarnings("ignore: Regressors info might be incorrect.")
@pytest.mark.parametrize(
    "in_column_regressor, out_column, expected_regressors",
    [
        (True, "output_regressor", ["output_regressor"]),
        (False, "output_regressor", []),
    ],
)
def test_get_regressors_info_le(
    ts_for_label_encoding, in_column_regressor, out_column, expected_regressors, in_column="regressor_0"
):
    ts, _ = ts_for_label_encoding
    if in_column_regressor:
        ts._regressors.append(in_column)
    else:
        ts._regressors.remove(in_column)
    le = LabelEncoderTransform(in_column=in_column, out_column=out_column)
    le.fit(ts)
    regressors_info = le.get_regressors_info()
    assert sorted(regressors_info) == sorted(expected_regressors)


@pytest.mark.filterwarnings("ignore: Regressors info might be incorrect.")
@pytest.mark.parametrize(
    "in_column_regressor, out_column, expected_regressors",
    [
        (True, "output_regressor", ["output_regressor_0", "output_regressor_1"]),
        (False, "output_regressor", []),
    ],
)
def test_get_regressors_info_ohe(
    ts_for_ohe_encoding, in_column_regressor, out_column, expected_regressors, in_column="regressor_0"
):
    ts, _ = ts_for_ohe_encoding
    if in_column_regressor:
        ts._regressors.append(in_column)
    else:
        ts._regressors.remove(in_column)
    ohe = OneHotEncoderTransform(in_column=in_column, out_column=out_column)
    ohe.fit(ts)
    regressors_info = ohe.get_regressors_info()
    assert sorted(regressors_info) == sorted(expected_regressors)


@pytest.mark.parametrize("dtype", ["float", "int", "str", "category"])
def test_save_load_le(dtype):
    ts, answers = get_ts_for_label_encoding(dtype=dtype)
    for i in range(3):
        transform = LabelEncoderTransform(in_column=f"regressor_{i}", out_column="test")
        assert_transformation_equals_loaded_original(transform=transform, ts=ts)


@pytest.mark.parametrize("dtype", ["float", "int", "str", "category"])
def test_save_load_ohe(dtype):
    ts, answers = get_ts_for_ohe_encoding(dtype=dtype)
    for i in range(3):
        transform = OneHotEncoderTransform(in_column=f"regressor_{i}", out_column="test")
        assert_transformation_equals_loaded_original(transform=transform, ts=ts)


@pytest.mark.parametrize(
    "transform", (OneHotEncoderTransform(in_column="regressor_0"), LabelEncoderTransform(in_column="regressor_0"))
)
def test_get_regressors_info_not_fitted(transform):
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


def test_params_to_tune_ohe():
    transform = OneHotEncoderTransform(in_column="regressor_0")
    assert len(transform.params_to_tune()) == 0


def test_params_to_tune_label_encoder(ts_for_label_encoding):
    ts, _ = ts_for_label_encoding
    for i in range(3):
        transform = LabelEncoderTransform(in_column=f"regressor_{i}", out_column="test")
        assert len(transform.params_to_tune()) > 0
        assert_sampling_is_valid(transform=transform, ts=ts)
