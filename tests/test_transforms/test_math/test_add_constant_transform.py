import numpy as np
import pytest

from etna.datasets import TSDataset
from etna.transforms.math import AddConstTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.fixture()
def example_ts_(example_df_) -> TSDataset:
    ts = TSDataset(example_df_, freq="D")
    return ts


@pytest.mark.parametrize("value", (-3.14, 6, 9.99))
def test_addconstpreproc_value(example_ts_: TSDataset, value: float):
    """Check the value of transform result"""
    preprocess = AddConstTransform(in_column="target", value=value, inplace=True)
    example_df_ = example_ts_.to_pandas()
    result = preprocess.fit_transform(ts=example_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result[segment]["target"], example_df_[segment]["target_no_change"] + value
        )


@pytest.mark.parametrize("out_column", (None, "result"))
def test_addconstpreproc_out_column_naming(example_ts_: TSDataset, out_column: str):
    """Check generated name of new column"""
    preprocess = AddConstTransform(in_column="target", value=4.2, inplace=False, out_column=out_column)
    result = preprocess.fit_transform(ts=example_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        if out_column:
            assert out_column in result[segment]
        else:
            assert preprocess.__repr__() in result[segment]


def test_addconstpreproc_value_out_column(example_ts_: TSDataset):
    """Check the value of transform result in case of given out column"""
    out_column = "result"
    preprocess = AddConstTransform(in_column="target", value=5.5, inplace=False, out_column=out_column)
    example_df_ = example_ts_.to_pandas()
    result = preprocess.fit_transform(ts=example_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            result[segment][out_column], example_df_[segment]["target_no_change"] + 5.5
        )


@pytest.mark.parametrize("value", (-5, 3.14, 33))
def test_inverse_transform(example_ts_: TSDataset, value: float):
    """Check that inverse_transform rolls back transform result"""
    preprocess = AddConstTransform(in_column="target", value=value)
    example_df_ = example_ts_.to_pandas()
    preprocess.fit_transform(ts=example_ts_)
    preprocess.inverse_transform(ts=example_ts_)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            example_ts_.to_pandas()[segment]["target"], example_df_[segment]["target_no_change"]
        )


def test_inverse_transform_out_column(example_ts_: TSDataset):
    """Check that inverse_transform rolls back transform result in case of given out_column"""
    out_column = "test"
    preprocess = AddConstTransform(in_column="target", value=10.1, inplace=False, out_column=out_column)
    preprocess.fit_transform(ts=example_ts_)
    inversed = preprocess.inverse_transform(ts=example_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        assert out_column in inversed[segment]


def test_fit_transform_with_nans(ts_diff_endings):
    transform = AddConstTransform(in_column="target", value=10)
    transform.fit_transform(ts_diff_endings)


@pytest.mark.parametrize("inplace", [False, True])
def test_save_load(inplace, example_tsds):
    transform = AddConstTransform(in_column="target", value=10, inplace=inplace)
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)


def test_get_regressors_info_not_fitted():
    transform = AddConstTransform(in_column="target", value=1, out_column="out_column")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


def test_params_to_tune():
    transform = AddConstTransform(in_column="target", value=1)
    assert len(transform.params_to_tune()) == 0
