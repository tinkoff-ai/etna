import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.math import AddConstTransform


@pytest.mark.parametrize("value", (-3.14, 6, 9.99))
def test_addconstpreproc_value(example_df_: pd.DataFrame, value: float):
    """Check the value of transform result"""
    ts_example = TSDataset(example_df_, freq="D")
    preprocess = AddConstTransform(in_column="target", value=value, inplace=True)
    result = preprocess.fit_transform(ts=ts_example)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result.df[segment]["target"], example_df_[segment]["target_no_change"] + value
        )


@pytest.mark.parametrize("out_column", (None, "result"))
def test_addconstpreproc_out_column_naming(example_df_: pd.DataFrame, out_column: str):
    """Check generated name of new column"""
    ts_example = TSDataset(example_df_, freq="D")
    preprocess = AddConstTransform(in_column="target", value=4.2, inplace=False, out_column=out_column)
    result = preprocess.fit_transform(ts=ts_example)
    for segment in ["segment_1", "segment_2"]:
        if out_column:
            assert out_column in result.df[segment]
        else:
            assert preprocess.__repr__() in result.df[segment]


def test_addconstpreproc_value_out_column(example_df_: pd.DataFrame):
    """Check the value of transform result in case of given out column"""
    ts_example = TSDataset(example_df_, freq="D")
    out_column = "result"
    preprocess = AddConstTransform(in_column="target", value=5.5, inplace=False, out_column=out_column)
    result = preprocess.fit_transform(ts=ts_example)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result.df[segment][out_column], example_df_[segment]["target_no_change"] + 5.5
        )


@pytest.mark.parametrize("value", (-5, 3.14, 33))
def test_inverse_transform(example_df_: pd.DataFrame, value: float):
    """Check that inverse_transform rolls back transform result"""
    ts_example = TSDataset(example_df_.copy(), freq="D")
    preprocess = AddConstTransform(in_column="target", value=value)
    transformed_target = preprocess.fit_transform(ts=ts_example)
    inversed = preprocess.inverse_transform(ts=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(inversed.df[segment]["target"], example_df_[segment]["target_no_change"])


def test_inverse_transform_out_column(example_df_: pd.DataFrame):
    """Check that inverse_transform rolls back transform result in case of given out_column"""
    ts_example = TSDataset(example_df_.copy(), freq="D")
    out_column = "test"
    preprocess = AddConstTransform(in_column="target", value=10.1, inplace=False, out_column=out_column)
    transformed_target = preprocess.fit_transform(ts=ts_example)
    inversed = preprocess.inverse_transform(ts=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        assert out_column in inversed.df[segment]


@pytest.mark.xfail(reason="TSDataset 2.0")
def test_fit_transform_with_nans(ts_diff_endings):
    transform = AddConstTransform(in_column="target", value=10)
    ts_diff_endings.fit_transform([transform])
