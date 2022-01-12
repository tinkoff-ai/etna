import numpy as np
import pytest

from etna.datasets import TSDataset
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
def df_for_categorical_encoding():
    return TSDataset.to_dataset(
        generate_periodic_df(periods=4, start_time="2020-01-01", scale=10, period=3, n_segments=2)
    )


def test_label_encoder(df_for_categorical_encoding):
    """Test LabelEncoderTransform correct works."""
    le = LabelEncoderTransform(in_column="target")
    le.fit(df_for_categorical_encoding)
    expected_values = np.array([[0, 1], [1, 0], [2, 0], [0, 1]])
    np.testing.assert_array_almost_equal(le.transform(df_for_categorical_encoding).values, expected_values)


@pytest.mark.parametrize(
    "strategy, expected_values",
    [
        ("new_value", np.array([[0, 0], [1, -1], [-1, -1]])),
        ("None", np.array([[0, 0], [1, np.nan], [np.nan, np.nan]])),
        ("mean", np.array([[0, 0], [1, 0], [0.5, 0]])),
    ],
)
def test_new_value_label(two_df_with_new_values, strategy, expected_values):
    """Test LabelEncoderTransform correct works with unknown values."""
    df1, df2 = two_df_with_new_values
    le = LabelEncoderTransform(in_column="target", strategy=strategy)
    le.fit(df1)
    np.testing.assert_array_almost_equal(le.transform(df2).values, expected_values)


def test_value_error_label(df_for_categorical_encoding):
    """Test LabelEncoderTransform with wrong strategy."""
    with pytest.raises(ValueError, match="The strategy"):
        le = LabelEncoderTransform(in_column="target", strategy="new_vlue")
        le.fit(df_for_categorical_encoding)
        le.transform(df_for_categorical_encoding)


def test_ohe(df_for_categorical_encoding):
    """Test OneHotEncoderTransform correct works."""
    ohe = OneHotEncoderTransform(in_column="target")
    ohe.fit(df_for_categorical_encoding)
    expected_values = np.array(
        [
            [1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 5.0],
            [0.0, 1.0, 0.0, 8.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 9.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 5.0],
        ]
    )
    np.testing.assert_array_almost_equal(ohe.transform(df_for_categorical_encoding).values, expected_values)


def test_new_value_ohe(two_df_with_new_values):
    """Test OneHotEncoderTransform correct works with unknown values."""
    expected_values = np.array(
        [[5.0, 1.0, 0.0, 5.0, 1.0, 0.0], [8.0, 0.0, 1.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    df1, df2 = two_df_with_new_values
    ohe = OneHotEncoderTransform(in_column="target", out_column="targets")
    ohe.fit(df1)
    np.testing.assert_array_almost_equal(ohe.transform(df2).values, expected_values)


def test_naming_ohe(two_df_with_new_values):
    """Test OneHotEncoderTransform gives the correct columns."""
    df1, df2 = two_df_with_new_values
    ohe = OneHotEncoderTransform(in_column="target", out_column="targets")
    ohe.fit(df1)
    segments = ["segment_0", "segment_1"]
    target = ["target", "targets_0", "targets_1"]
    assert set([(i, j) for i in segments for j in target]) == set(ohe.transform(df2).columns.values)
