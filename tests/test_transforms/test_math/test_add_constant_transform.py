import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.math import AddConstTransform


@pytest.mark.parametrize("value", (-3.14, 6, 9.99))
def test_addconstpreproc_value(example_df_: pd.DataFrame, value: float):
    """Check the value of transform result"""
    preprocess = AddConstTransform(in_column="target", value=value, inplace=True)
    ts = TSDataset(example_df_, freq="D")
    result = preprocess.fit_transform(ts=ts)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result.df[segment]["target"], example_df_[segment]["target_no_change"] + value
        )


@pytest.mark.parametrize("out_column", (None, "result"))
def test_addconstpreproc_out_column_naming(example_df_: pd.DataFrame, out_column: str):
    """Check generated name of new column"""
    preprocess = AddConstTransform(in_column="target", value=4.2, inplace=False, out_column=out_column)
    ts = TSDataset(example_df_, freq="D")
    result = preprocess.fit_transform(ts=ts)
    for segment in ["segment_1", "segment_2"]:
        if out_column:
            assert out_column in result.df[segment]
        else:
            assert preprocess.__repr__() in result.df[segment]


def test_addconstpreproc_value_out_column(example_df_: pd.DataFrame):
    """Check the value of transform result in case of given out column"""
    out_column = "result"
    preprocess = AddConstTransform(in_column="target", value=5.5, inplace=False, out_column=out_column)
    ts = TSDataset(example_df_, freq="D")
    result = preprocess.fit_transform(ts=ts)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result.df[segment][out_column], example_df_[segment]["target_no_change"] + 5.5
        )


@pytest.mark.parametrize("value", (-5, 3.14, 33))
def test_inverse_transform(example_df_: pd.DataFrame, value: float):
    """Check that inverse_transform rolls back transform result"""
    preprocess = AddConstTransform(in_column="target", value=value)
    ts = TSDataset(example_df_.copy(), freq="D")
    transformed_target = preprocess.fit_transform(ts=ts)
    inversed = preprocess.inverse_transform(ts=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(inversed.df[segment]["target"], example_df_[segment]["target_no_change"])


def test_inverse_transform_out_column(example_df_: pd.DataFrame):
    """Check that inverse_transform rolls back transform result in case of given out_column"""
    out_column = "test"
    preprocess = AddConstTransform(in_column="target", value=10.1, inplace=False, out_column=out_column)
    ts = TSDataset(example_df_, freq="D")
    transformed_target = preprocess.fit_transform(ts=ts)
    inversed = preprocess.inverse_transform(ts=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        assert out_column in inversed.df[segment]


def test_fit_transform_with_nans(ts_diff_endings):
    transform = AddConstTransform(in_column="target", value=10)
    ts_diff_endings = transform.fit_transform(ts=ts_diff_endings)
