from copy import deepcopy
from math import e

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms import AddConstTransform
from etna.transforms.math import LogTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.fixture
def non_positive_ts_(random_seed) -> TSDataset:
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
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def positive_ts_(random_seed) -> TSDataset:
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
    ts = TSDataset(df, freq="D")
    return ts


def test_negative_series_behavior(non_positive_ts_: TSDataset):
    """Check LogTransform behavior in case of negative-value series."""
    preprocess = LogTransform(in_column="target")
    with pytest.raises(ValueError):
        _ = preprocess.fit_transform(ts=non_positive_ts_)


def test_logpreproc_value(positive_ts_: TSDataset):
    """Check the value of transform result."""
    preprocess = LogTransform(in_column="target", base=10)
    value = preprocess.fit_transform(ts=positive_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(value[segment]["target"], positive_ts_.to_pandas()[segment]["expected"])


@pytest.mark.parametrize("out_column", (None, "log_transform"))
def test_logpreproc_noninplace_interface(positive_ts_: TSDataset, out_column: str):
    """Check the column name after non inplace transform."""
    preprocess = LogTransform(in_column="target", out_column=out_column, base=10, inplace=False)
    value = preprocess.fit_transform(ts=positive_ts_).to_pandas()
    expected_out_column = out_column if out_column is not None else preprocess.__repr__()
    for segment in ["segment_1", "segment_2"]:
        assert expected_out_column in value[segment]


def test_logpreproc_value_out_column(positive_ts_: TSDataset):
    """Check the value of transform result in case of given out column."""
    out_column = "target_log_10"
    preprocess = LogTransform(in_column="target", out_column=out_column, base=10, inplace=False)
    value = preprocess.fit_transform(ts=deepcopy(positive_ts_)).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(value[segment][out_column], positive_ts_.to_pandas()[segment]["expected"])


@pytest.mark.parametrize("base", (5, 10, e))
def test_inverse_transform(positive_ts_: TSDataset, base: int):
    """Check that inverse_transform rolls back transform result."""
    preprocess = LogTransform(in_column="target", base=base)
    positive_df_ = positive_ts_.to_pandas()
    preprocess.fit_transform(ts=positive_ts_)
    preprocess.inverse_transform(ts=positive_ts_)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            positive_ts_.to_pandas()[segment]["target"], positive_df_[segment]["target"]
        )


def test_inverse_transform_out_column(positive_ts_: TSDataset):
    """Check that inverse_transform rolls back transform result in case of given out_column."""
    out_column = "target_log_10"
    preprocess = LogTransform(in_column="target", out_column=out_column, base=10, inplace=False)
    preprocess.fit_transform(ts=positive_ts_)
    inversed = preprocess.inverse_transform(ts=positive_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        assert out_column in inversed[segment]


def test_fit_transform_with_nans(ts_diff_endings):
    transform = LogTransform(in_column="target", inplace=True)
    ts_diff_endings.fit_transform([AddConstTransform(in_column="target", value=100)] + [transform])


@pytest.mark.parametrize("inplace", [False, True])
def test_save_load(inplace, positive_ts_):
    ts = positive_ts_
    transform = LogTransform(in_column="target", inplace=inplace)
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


def test_get_regressors_info_not_fitted():
    transform = LogTransform(in_column="target")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


def test_params_to_tune():
    transform = LogTransform(in_column="target")
    assert len(transform.params_to_tune()) == 0
