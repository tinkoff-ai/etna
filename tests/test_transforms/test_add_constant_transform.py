import numpy as np
import pandas as pd
import pytest

from etna.transforms.add_constant import AddConstTransform


@pytest.mark.parametrize("value", (-3.14, 6, 9.99))
def test_addconstpreproc_value(example_df_: pd.DataFrame, value: float):
    """Check the value of transform result"""
    preprocess = AddConstTransform(in_column="target", value=value)
    result = preprocess.fit_transform(df=example_df_)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result[segment]["target"], example_df_[segment]["target_no_change"] + value
        )


def test_addconstpreproc_value_out_column(example_df_: pd.DataFrame):
    """Check the value of transform result in case of given out column"""
    expected_out_column = "target_add_5.5"
    preprocess = AddConstTransform(in_column="target", value=5.5, inplace=False)
    result = preprocess.fit_transform(df=example_df_)
    for segment in ["segment_1", "segment_2"]:
        assert expected_out_column in result[segment]
        np.testing.assert_array_almost_equal(
            result[segment][expected_out_column], example_df_[segment]["target_no_change"] + 5.5
        )


@pytest.mark.parametrize("value", (-5, 3.14, 33))
def test_inverse_transform(example_df_: pd.DataFrame, value: float):
    """Check that inverse_transform rolls back transform result"""
    preprocess = AddConstTransform(in_column="target", value=value)
    transformed_target = preprocess.fit_transform(df=example_df_.copy())
    inversed = preprocess.inverse_transform(df=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(inversed[segment]["target"], example_df_[segment]["target_no_change"])


def test_inverse_transform_out_column(example_df_: pd.DataFrame):
    """Check that inverse_transform rolls back transform result in case of given out_column"""
    expected_out_column = "target_add_10.1"
    preprocess = AddConstTransform(in_column="target", value=10.1, inplace=False)
    transformed_target = preprocess.fit_transform(df=example_df_)
    inversed = preprocess.inverse_transform(df=transformed_target)
    for segment in ["segment_1", "segment_2"]:
        assert expected_out_column in inversed[segment]
