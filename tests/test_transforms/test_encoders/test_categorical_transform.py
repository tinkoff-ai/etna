import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_periodic_df
from etna.transforms.encoders import categorical
from etna.transforms.encoders.categorical import LabelEncoderTransform


@pytest.fixture
def two_df_with_new_values():
    df1 = TSDataset.to_dataset(generate_periodic_df(3, "2020-01-01", 10, 2, n_segments=2))
    df2 = TSDataset.to_dataset(generate_periodic_df(3, "2020-01-01", 10, 3, n_segments=2))
    return df1, df2


@pytest.fixture
def df_for_categorical_encoding():
    return TSDataset.to_dataset(generate_periodic_df(4, "2020-01-01", 10, 3, n_segments=2))


def test_label_encoder(df_for_categorical_encoding):
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
    df1, df2 = two_df_with_new_values
    le = LabelEncoderTransform(in_column="target", strategy=strategy)
    le.fit(df1)
    np.testing.assert_array_almost_equal(le.transform(df2).values, expected_values)


def test_value_error_label(df_for_categorical_encoding):
    with pytest.raises(ValueError, match="There are no"):
        le = LabelEncoderTransform(in_column="target", strategy="new_vlue")
        le.fit(df_for_categorical_encoding)
        le.transform(df_for_categorical_encoding)
