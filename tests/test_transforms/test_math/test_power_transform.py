from copy import deepcopy
from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from etna.datasets import TSDataset
from etna.transforms import AddConstTransform
from etna.transforms.math import BoxCoxTransform
from etna.transforms.math import YeoJohnsonTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.fixture
def non_positive_ts() -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_1["segment"] = "Moscow"
    df_1["target"] = 0
    df_1["exog"] = -1
    df_2["segment"] = "Omsk"
    df_2["target"] = -1
    df_2["exog"] = -7
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="1d")
    return ts


@pytest.fixture
def positive_ts() -> TSDataset:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    generator = np.random.RandomState(seed=1)
    df_1["segment"] = "Moscow"
    df_1["target"] = np.abs(generator.normal(loc=10, scale=1, size=len(df_1))) + 1
    df_1["exog"] = np.abs(generator.normal(loc=15, scale=1, size=len(df_1))) + 1
    df_2["segment"] = "Omsk"
    df_2["target"] = np.abs(generator.normal(loc=20, scale=1, size=len(df_2))) + 1
    df_2["exog"] = np.abs(generator.normal(loc=4, scale=1, size=len(df_2))) + 1
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df, freq="1d")
    return ts


@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_non_positive_series_behavior(non_positive_ts: TSDataset, mode: str):
    """Check BoxCoxPreprocessing behavior in case of negative-value series."""
    preprocess = BoxCoxTransform(mode=mode)
    with pytest.raises(ValueError):
        _ = preprocess.fit_transform(ts=non_positive_ts)


@pytest.mark.parametrize(
    "preprocessing_class,method", ((BoxCoxTransform, "box-cox"), (YeoJohnsonTransform, "yeo-johnson"))
)
def test_transform_value_all_columns(positive_ts: TSDataset, preprocessing_class: Any, method: str):
    """Check the value of transform result for all columns."""
    preprocess_none = preprocessing_class()
    original_df = positive_ts.to_pandas()
    preprocess_all = preprocessing_class(in_column=original_df.columns.get_level_values("feature").unique())

    value_none = preprocess_none.fit_transform(ts=deepcopy(positive_ts)).to_pandas()
    value_all = preprocess_all.fit_transform(ts=deepcopy(positive_ts)).to_pandas()
    true_values = PowerTransformer(method=method).fit_transform(original_df.values)
    npt.assert_array_almost_equal(value_none.values, true_values)
    npt.assert_array_almost_equal(value_all.values, true_values)


@pytest.mark.parametrize(
    "preprocessing_class,method", ((BoxCoxTransform, "box-cox"), (YeoJohnsonTransform, "yeo-johnson"))
)
def test_transform_value_one_column(positive_ts: TSDataset, preprocessing_class: Any, method: str):
    """Check the value of transform result."""
    preprocess = preprocessing_class(in_column="target")
    original_df = positive_ts.to_pandas()
    processed_values = preprocess.fit_transform(ts=positive_ts).to_pandas()
    target_processed_values = processed_values.loc[:, pd.IndexSlice[:, "target"]].values
    rest_processed_values = processed_values.drop("target", axis=1, level="feature").values
    untouched_values = original_df.drop("target", axis=1, level="feature").values
    true_values = PowerTransformer(method=method).fit_transform(original_df.loc[:, pd.IndexSlice[:, "target"]].values)
    npt.assert_array_almost_equal(target_processed_values, true_values)
    npt.assert_array_almost_equal(rest_processed_values, untouched_values)


@pytest.mark.parametrize("preprocessing_class", (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_inverse_transform_all_columns(positive_ts: TSDataset, preprocessing_class: Any, mode: str):
    """Check that inverse_transform rolls back transform result for all columns."""
    preprocess_none = preprocessing_class(mode=mode)
    original_df = positive_ts.to_pandas()
    preprocess_all = preprocessing_class(in_column=original_df.columns.get_level_values("feature").unique(), mode=mode)
    transformed_target_none = preprocess_none.fit_transform(ts=deepcopy(positive_ts))
    transformed_target_all = preprocess_all.fit_transform(ts=deepcopy(positive_ts))
    inversed_target_none = preprocess_none.inverse_transform(ts=transformed_target_none).to_pandas()
    inversed_target_all = preprocess_all.inverse_transform(ts=transformed_target_all).to_pandas()
    np.testing.assert_array_almost_equal(inversed_target_none.values, original_df.values)
    np.testing.assert_array_almost_equal(inversed_target_all.values, original_df.values)


@pytest.mark.parametrize("preprocessing_class", (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_inverse_transform_one_column(positive_ts: TSDataset, preprocessing_class: Any, mode: str):
    """Check that inverse_transform rolls back transform result for one column."""
    preprocess = preprocessing_class(in_column="target", mode=mode)
    original_df = positive_ts.to_pandas()
    transformed_target = preprocess.fit_transform(ts=positive_ts)
    inversed_target = preprocess.inverse_transform(ts=transformed_target).to_pandas()
    np.testing.assert_array_almost_equal(inversed_target.values, original_df.values)


@pytest.mark.parametrize("preprocessing_class", (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_fit_transform_with_nans(preprocessing_class, mode, ts_diff_endings):
    preprocess = preprocessing_class(in_column="target", mode=mode)
    add_const = AddConstTransform(in_column="target", value=100)
    ts_diff_endings = add_const.fit_transform(ts_diff_endings)
    preprocess.fit_transform(ts_diff_endings)


@pytest.mark.parametrize("transform_constructor", (BoxCoxTransform, YeoJohnsonTransform))
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_save_load(transform_constructor, mode, positive_ts):
    ts = positive_ts
    transform = transform_constructor(in_column="target", mode=mode)
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


@pytest.mark.parametrize("transform_constructor, expected_length", [(BoxCoxTransform, 2), (YeoJohnsonTransform, 2)])
@pytest.mark.parametrize("mode", ("macro", "per-segment"))
def test_params_to_tune(transform_constructor, expected_length, mode, positive_ts):
    ts = positive_ts
    transform = transform_constructor(in_column="target", mode=mode)
    assert len(transform.params_to_tune()) == expected_length
    assert_sampling_is_valid(transform=transform, ts=ts)
