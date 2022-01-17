import numpy as np
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


@pytest.fixture
def two_df_with_new_values():
    df1 = TSDataset.to_dataset(
        generate_periodic_df(periods=3, start_time="2020-01-01", scale=10, period=2, n_segments=2)
    )
    df2 = TSDataset.to_dataset(
        generate_periodic_df(periods=3, start_time="2020-01-01", scale=10, period=3, n_segments=2)
    )
    return df1, df2


@pytest.fixture
def df_for_ohe_encoding():
    df_to_forecast = generate_ar_df(10, start_time="2021-01-01", n_segments=1)
    df_regressors = generate_periodic_df(12, start_time="2021-01-01", scale=10, period=2, n_segments=5)
    df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    df_regressors.columns = ["timestamp"] + [f"regressor_{i}" for i in range(5)]
    df_regressors["segment"] = "segment_0"
    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors)

    regressor_0 = tsdataset.df.copy()["segment_0"]
    regressor_0["test_0"] = regressor_0["regressor_0"].apply(lambda x: float(x == 5))
    regressor_0["test_1"] = regressor_0["regressor_0"].apply(lambda x: float(x == 8))
    regressor_0["test_0"] = regressor_0["test_0"].astype("category")
    regressor_0["test_1"] = regressor_0["test_1"].astype("category")

    regressor_1 = tsdataset.df.copy()["segment_0"]
    regressor_1["test_0"] = regressor_1["regressor_1"].apply(lambda x: float(x == 5))
    regressor_1["test_1"] = regressor_1["regressor_1"].apply(lambda x: float(x == 9))
    regressor_1["test_0"] = regressor_1["test_0"].astype("category")
    regressor_1["test_1"] = regressor_1["test_1"].astype("category")

    regressor_2 = tsdataset.df.copy()["segment_0"]
    regressor_2["test_0"] = regressor_2["regressor_2"].apply(lambda x: float(x == 0))
    regressor_2["test_0"] = regressor_2["test_0"].astype("category")

    return tsdataset.df, (regressor_0, regressor_1, regressor_2)


@pytest.fixture
def df_for_label_encoding():
    df_to_forecast = generate_ar_df(10, start_time="2021-01-01", n_segments=1)
    df_regressors = generate_periodic_df(12, start_time="2021-01-01", scale=10, period=2, n_segments=5)
    df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    df_regressors.columns = ["timestamp"] + [f"regressor_{i}" for i in range(5)]
    df_regressors["segment"] = "segment_0"
    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors)

    regressor_0 = tsdataset.df.copy()["segment_0"]
    regressor_0["test"] = regressor_0["regressor_0"].apply(lambda x: float(x == 8))
    regressor_0["test"] = regressor_0["test"].astype("category")

    regressor_1 = tsdataset.df.copy()["segment_0"]
    regressor_1["test"] = regressor_1["regressor_1"].apply(lambda x: float(x == 9))
    regressor_1["test"] = regressor_1["test"].astype("category")

    regressor_2 = tsdataset.df.copy()["segment_0"]
    regressor_2["test"] = regressor_2["regressor_2"].apply(lambda x: float(x == 1))
    regressor_2["test"] = regressor_2["test"].astype("category")

    return tsdataset.df, (regressor_0, regressor_1, regressor_2)


@pytest.fixture
def df_for_naming():
    df_to_forecast = generate_ar_df(10, start_time="2021-01-01", n_segments=1)
    df_regressors = generate_periodic_df(12, start_time="2021-01-01", scale=10, period=2, n_segments=2)
    df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    df_regressors.columns = ["timestamp"] + ["regressor_1", "2"]
    df_regressors["segment"] = "segment_0"
    df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    df_regressors = TSDataset.to_dataset(df_regressors)
    tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors)
    return tsdataset.df


def test_label_encoder_simple(df_for_label_encoding):
    """Test that LabelEncoderTransform works correct in a simple cases."""
    df, answers = df_for_label_encoding
    for i in range(3):
        le = LabelEncoderTransform(in_column=f"regressor_{i}", inplace=False, out_column="test")
        le.fit(df)
        cols = le.transform(df)["segment_0"].columns
        assert le.transform(df)["segment_0"][cols].equals(answers[i][cols])


def test_ohe_encoder_simple(df_for_ohe_encoding):
    """Test that OneHotEncoderTransform works correct in a simple case."""
    df, answers = df_for_ohe_encoding
    for i in range(3):
        ohe = OneHotEncoderTransform(in_column=f"regressor_{i}", out_column="test")
        ohe.fit(df)
        cols = ohe.transform(df)["segment_0"].columns
        assert ohe.transform(df)["segment_0"][cols].equals(answers[i][cols])


def test_value_error_label_encoder(df_for_label_encoding):
    """Test LabelEncoderTransform with wrong strategy."""
    df, _ = df_for_label_encoding
    with pytest.raises(ValueError, match="The strategy"):
        le = LabelEncoderTransform(in_column="target", strategy="new_vlue")
        le.fit(df)
        le.transform(df)


@pytest.mark.parametrize(
    "strategy, expected_values",
    [
        ("new_value", np.array([[0, 0], [1, -1], [-1, -1]])),
        ("none", np.array([[0, 0], [1, np.nan], [np.nan, np.nan]])),
        ("mean", np.array([[0, 0], [1, 0], [0.5, 0]])),
    ],
)
def test_new_value_label_encoder(two_df_with_new_values, strategy, expected_values):
    """Test LabelEncoderTransform correct works with unknown values."""
    df1, df2 = two_df_with_new_values
    le = LabelEncoderTransform(in_column="target", strategy=strategy)
    le.fit(df1)
    np.testing.assert_array_almost_equal(le.transform(df2).values, expected_values)


def test_new_value_ohe_encoder(two_df_with_new_values):
    """Test OneHotEncoderTransform correct works with unknown values."""
    expected_values = np.array(
        [[5.0, 1.0, 0.0, 5.0, 1.0, 0.0], [8.0, 0.0, 1.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    df1, df2 = two_df_with_new_values
    ohe = OneHotEncoderTransform(in_column="target", out_column="targets")
    ohe.fit(df1)
    np.testing.assert_array_almost_equal(ohe.transform(df2).values, expected_values)


def test_naming_ohe_encoder(two_df_with_new_values):
    """Test OneHotEncoderTransform gives the correct columns."""
    df1, df2 = two_df_with_new_values
    ohe = OneHotEncoderTransform(in_column="target", out_column="targets")
    ohe.fit(df1)
    segments = ["segment_0", "segment_1"]
    target = ["target", "targets_0", "targets_1"]
    assert set([(i, j) for i in segments for j in target]) == set(ohe.transform(df2).columns.values)


@pytest.mark.parametrize(
    "in_column, prefix",
    [("2", ""), ("regressor_1", "regressor_")],
)
def test_naming_ohe_encoder_no_out_column(df_for_naming, in_column, prefix):
    """Test OneHotEncoderTransform gives the correct columns with no out_column."""
    df = df_for_naming
    ohe = OneHotEncoderTransform(in_column=in_column)
    ohe.fit(df)
    answer = set(
        list(df["segment_0"].columns) + [prefix + str(ohe.__repr__()) + "_0", prefix + str(ohe.__repr__()) + "_1"]
    )
    assert answer == set(ohe.transform(df)["segment_0"].columns.values)


@pytest.mark.parametrize(
    "in_column, prefix",
    [("2", ""), ("regressor_1", "regressor_")],
)
def test_naming_label_encoder_no_out_column(df_for_naming, in_column, prefix):
    """Test LabelEncoderTransform gives the correct columns with no out_column."""
    df = df_for_naming
    le = LabelEncoderTransform(in_column=in_column, inplace=False)
    le.fit(df)
    answer = set(list(df["segment_0"].columns) + [prefix + str(le.__repr__())])
    assert answer == set(le.transform(df)["segment_0"].columns.values)


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
        return x ** 2 + rng.normal(0, 0.01)

    df_to_forecast["segment_0", "target"] = df_regressors["segment_0"]["regressor_0"][:100].apply(f)
    ts = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors)
    return ts


def test_ohe_sanity(ts_for_ohe_sanity):
    """Test for correct work in the full forecasting pipeline."""
    HORIZON = 10
    train_ts, test_ts = ts_for_ohe_sanity.train_test_split(test_size=HORIZON)
    ohe = OneHotEncoderTransform(in_column="regressor_0")
    filt = FilterFeaturesTransform(exclude=["regressor_0"])
    train_ts.fit_transform([ohe, filt])
    model = LinearPerSegmentModel()
    model.fit(train_ts)
    future_ts = train_ts.make_future(HORIZON)
    forecast_ts = model.forecast(future_ts)
    r2 = R2()
    assert 1 - r2(test_ts, forecast_ts)["segment_0"] < 1e-5
