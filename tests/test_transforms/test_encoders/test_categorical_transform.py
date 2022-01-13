import numpy as np
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.datasets import generate_periodic_df
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
    regressor_0["test_0"] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    regressor_0["test_1"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    regressor_0["test_0"] = regressor_0["test_0"].astype("category")
    regressor_0["test_1"] = regressor_0["test_1"].astype("category")

    regressor_1 = tsdataset.df.copy()["segment_0"]
    regressor_1["test_0"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    regressor_1["test_1"] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    regressor_1["test_0"] = regressor_1["test_0"].astype("category")
    regressor_1["test_1"] = regressor_1["test_1"].astype("category")

    regressor_2 = tsdataset.df.copy()["segment_0"]
    regressor_2["test_0"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
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
    regressor_0["test"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    regressor_0["test"] = regressor_0["test"].astype("category")

    regressor_1 = tsdataset.df.copy()["segment_0"]
    regressor_1["test"] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    regressor_1["test"] = regressor_1["test"].astype("category")

    regressor_2 = tsdataset.df.copy()["segment_0"]
    regressor_2["test"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    regressor_2["test"] = regressor_2["test"].astype("category")

    return tsdataset.df, (regressor_0, regressor_1, regressor_2)


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
