from typing import List
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from etna.datasets.tsdataset import TSDataset
from etna.transforms.math import LagTransform


@pytest.fixture
def int_df_one_segment() -> pd.DataFrame:
    """Generate dataframe with simple targets for lags check."""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-06-01")})
    df["segment"] = "segment_1"
    df["target"] = np.arange(0, len(df))
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def int_df_two_segments(int_df_one_segment) -> pd.DataFrame:
    """Generate dataframe with simple targets for lags check."""
    df_1 = int_df_one_segment.reset_index()
    df_2 = int_df_one_segment.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(df)


def test_repr():
    """Test that that __repr__ method works fine."""
    transform_class = "LagTransform"
    lags = list(range(8, 24, 1))
    out_column = "lag_feature"

    transform_without_out_column = LagTransform(lags=lags, in_column="target")
    transform_with_out_column = LagTransform(lags=lags, in_column="target", out_column=out_column)

    true_repr_out_column = f"{transform_class}(in_column = 'target', lags = {lags}, out_column = '{out_column}', )"
    true_repr_no_out_column = f"{transform_class}(in_column = 'target', lags = {lags}, out_column = None, )"

    no_out_column_repr = transform_without_out_column.__repr__()
    out_column_repr = transform_with_out_column.__repr__()

    assert no_out_column_repr == true_repr_no_out_column
    assert out_column_repr == true_repr_out_column


@pytest.mark.parametrize(
    "lags,expected_columns",
    (
        (3, ["regressor_lag_feature_1", "regressor_lag_feature_2", "regressor_lag_feature_3"]),
        ([5, 8], ["regressor_lag_feature_5", "regressor_lag_feature_8"]),
    ),
)
def test_interface_two_segments_out_column(
    lags: Union[int, Sequence[int]], expected_columns: List[str], int_df_two_segments
):
    """Test that transform generates correct column names using out_column parameter."""
    lf = LagTransform(in_column="target", lags=lags, out_column="regressor_lag_feature")
    lags_df = lf.fit_transform(df=int_df_two_segments)
    for segment in lags_df.columns.get_level_values("segment").unique():
        lags_df_lags_columns = sorted(filter(lambda x: x.startswith("regressor_lag_feature"), lags_df[segment].columns))
        assert lags_df_lags_columns == expected_columns


@pytest.mark.parametrize("lags", (3, [5, 8]))
def test_interface_two_segments_repr(lags: Union[int, Sequence[int]], int_df_two_segments):
    """Test that transform generates correct column names without setting out_column parameter."""
    segments = int_df_two_segments.columns.get_level_values("segment").unique()
    transform = LagTransform(in_column="target", lags=lags)
    transformed_df = transform.fit_transform(int_df_two_segments)
    columns = transformed_df.columns.get_level_values("feature").unique().drop("target")
    assert len(columns) == len(lags) if isinstance(lags, list) else 1
    for column in columns:
        # check that a transform can be created from column name and it generates the same results
        transform_temp = eval(column)
        df_temp = transform_temp.fit_transform(int_df_two_segments)
        columns_temp = df_temp.columns.get_level_values("feature").unique().drop("target")
        assert len(columns_temp) == 1
        generated_column = columns_temp[0]
        assert generated_column == column
        assert df_temp.loc[:, pd.IndexSlice[segments, generated_column]].equals(
            transformed_df.loc[:, pd.IndexSlice[segments, column]]
        )


@pytest.mark.parametrize("lags", (12, [4, 6, 8, 16]))
def test_lags_values_two_segments(lags: Union[int, Sequence[int]], int_df_two_segments):
    """Test that transform generates correct values."""
    lf = LagTransform(in_column="target", lags=lags, out_column="regressor_lag_feature")
    lags_df = lf.fit_transform(df=int_df_two_segments)
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    for segment in lags_df.columns.get_level_values("segment").unique():
        for lag in lags:
            true_values = pd.Series([None] * lag + list(int_df_two_segments[segment, "target"].values[:-lag]))
            assert_almost_equal(true_values.values, lags_df[segment, f"regressor_lag_feature_{lag}"].values)


@pytest.mark.parametrize("lags", (0, -1, (10, 15, -2)))
def test_invalid_lags_value_two_segments(lags):
    """Test that LagTransform can't be created with non-positive lags."""
    with pytest.raises(ValueError):
        _ = LagTransform(in_column="target", lags=lags)


def test_fit_transform_with_nans(ts_diff_endings):
    """Test that transform correctly works with NaNs at the end."""
    transform = LagTransform(in_column="target", lags=10)
    ts_diff_endings.fit_transform([transform])
