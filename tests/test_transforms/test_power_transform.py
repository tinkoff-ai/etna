from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from etna.datasets import TSDataset
from etna.transforms.power import BoxCoxTransform
from etna.transforms.power import YeoJohnsonTransform


@pytest.fixture
def non_positive_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_1["segment"] = "Moscow"
    df_1["target"] = 0
    df_2["segment"] = "Omsk"
    df_2["target"] = -1
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)


@pytest.fixture
def positive_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    generator = np.random.RandomState(seed=1)
    df_1["segment"] = "Moscow"
    df_1["target"] = generator.normal(loc=10, scale=1, size=len(df_1))
    df_2["segment"] = "Omsk"
    df_2["target"] = generator.normal(loc=20, scale=1, size=len(df_1))
    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(classic_df)


def test_non_positive_series_behavior(non_positive_df: pd.DataFrame):
    """Check BoxCoxPreprocessing behavior in case of negative-value series."""
    preprocess = BoxCoxTransform()
    with pytest.raises(ValueError):
        _ = preprocess.fit_transform(df=non_positive_df)


@pytest.mark.parametrize(
    "preprocessing_class,method", ((BoxCoxTransform, "box-cox"), (YeoJohnsonTransform, "yeo-johnson"))
)
def test_transform_value_all_columns(positive_df: pd.DataFrame, preprocessing_class: Any, method: str):
    """Check the value of transform result for all columns."""
    preprocess_none = preprocessing_class()
    preprocess_all = preprocessing_class(in_column=positive_df.columns.get_level_values("feature").unique())
    value_none = preprocess_none.fit_transform(df=positive_df.copy())
    value_all = preprocess_all.fit_transform(df=positive_df.copy())
    true_values = PowerTransformer(method=method).fit_transform(positive_df.values)
    npt.assert_array_almost_equal(value_none.values, true_values)
    npt.assert_array_almost_equal(value_all.values, true_values)


@pytest.mark.parametrize(
    "preprocessing_class,method", ((BoxCoxTransform, "box-cox"), (YeoJohnsonTransform, "yeo-johnson"))
)
def test_transform_value_one_column(positive_df: pd.DataFrame, preprocessing_class: Any, method: str):
    """Check the value of transform result."""
    preprocess = preprocessing_class(in_column="target")
    value = preprocess.fit_transform(df=positive_df.copy())
    true_values = PowerTransformer(method=method).fit_transform(positive_df.loc[:, pd.IndexSlice[:, "target"]].values)
    npt.assert_array_almost_equal(value.values, true_values)


@pytest.mark.parametrize("preprocessing_class", (BoxCoxTransform, YeoJohnsonTransform))
def test_inverse_transform_all_columns(positive_df: pd.DataFrame, preprocessing_class: Any):
    """Check that inverse_transform rolls back transform result for all columns."""
    preprocess_none = preprocessing_class()
    preprocess_all = preprocessing_class(in_column=positive_df.columns.get_level_values("feature").unique())
    transformed_target_none = preprocess_none.fit_transform(df=positive_df.copy())
    transformed_target_all = preprocess_all.fit_transform(df=positive_df.copy())
    inversed_target_none = preprocess_none.inverse_transform(df=transformed_target_none)
    inversed_target_all = preprocess_none.inverse_transform(df=transformed_target_all)
    np.testing.assert_array_almost_equal(inversed_target_none.values, positive_df.values)
    np.testing.assert_array_almost_equal(inversed_target_all.values, positive_df.values)


@pytest.mark.parametrize("preprocessing_class", (BoxCoxTransform, YeoJohnsonTransform))
def test_inverse_transform_one_column(positive_df: pd.DataFrame, preprocessing_class: Any):
    """Check that inverse_transform rolls back transform result for one column."""
    preprocess = preprocessing_class(in_column="target")
    transformed_target = preprocess.fit_transform(df=positive_df.copy())
    inversed_target = preprocess.inverse_transform(df=transformed_target)
    np.testing.assert_array_almost_equal(inversed_target.values, positive_df.values)