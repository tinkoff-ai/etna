from math import e

import numpy as np
import pandas as pd
import pytest

from etna.transforms.log import LogTransform


@pytest.fixture
def non_positive_df_(random_seed) -> pd.DataFrame:
    """Generate dataset with non-positive target."""
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = ["segment_1"] * periods
    df1["target"] = np.random.uniform(-10, 0, size=periods)

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = ["segment_2"] * periods
    df2["target"] = np.random.uniform(0, 10, size=periods)

    df = pd.concat((df1, df2))
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


@pytest.fixture
def positive_df_(random_seed) -> pd.DataFrame:
    """Generate dataset with positive target."""
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = ["segment_1"] * periods
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1["expected"] = np.log10(df1["target"] + 1)

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = ["segment_2"] * periods
    df2["target"] = np.random.uniform(1, 15, size=periods)
    df2["expected"] = np.log10(df2["target"] + 1)

    df = pd.concat((df1, df2))
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


def test_negative_series_behavior(non_positive_df_: pd.DataFrame):
    """Check LogTransform behavior in case of negative-value series."""
    preprocess = LogTransform(in_column="target")
    with pytest.raises(ValueError):
        _ = preprocess.fit_transform(df=non_positive_df_)


def test_logpreproc_value(positive_df_: pd.DataFrame):
    """Check the value of transform result."""
    preprocess = LogTransform(in_column="target", base=10)
    value = preprocess.fit_transform(df=positive_df_)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(value[segment]["target"], positive_df_[segment]["expected"])


def test_logpreproc_value_out_column(positive_df_: pd.DataFrame):
    """Check the value of transform result in case of given out column."""
    expected_out_column = "target_log_10"
    preprocess = LogTransform(in_column="target", base=10, inplace=False)
    value = preprocess.fit_transform(df=positive_df_)
    for segment in ["segment_1", "segment_2"]:
        assert expected_out_column in value[segment]
        np.testing.assert_array_almost_equal(value[segment][expected_out_column], positive_df_[segment]["expected"])


@pytest.mark.parametrize("base", (5, 10, e))
def test_inverse_transform(positive_df_: pd.DataFrame, base: int):
    """Check that inverse_transform rolls back transform result."""
    preprocess = LogTransform(in_column="target", base=base)
    transformed_target = preprocess.fit_transform(df=positive_df_.copy())
    inversed = preprocess.inverse_transform(df=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(inversed[segment]["target"], positive_df_[segment]["target"])


def test_inverse_transform_out_column(positive_df_: pd.DataFrame):
    """Check that inverse_transform rolls back transform result in case of given out_column."""
    expected_out_column = "target_log_10"
    preprocess = LogTransform(in_column="target", base=10, inplace=False)
    transformed_target = preprocess.fit_transform(df=positive_df_)
    inversed = preprocess.inverse_transform(df=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        assert expected_out_column in inversed[segment]
