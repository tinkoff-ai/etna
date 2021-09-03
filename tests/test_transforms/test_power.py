from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer

from etna.transforms.power import BoxCoxTransform
from etna.transforms.power import YeoJohnsonTransform


@pytest.fixture
def non_positive_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    generator = np.random.RandomState(seed=1)
    df_1["segment"] = "Moscow"
    df_1["target"] = -1
    df_2["segment"] = "Omsk"
    df_2["target"] = -100
    classic_df = pd.concat([df_1, df_2], ignore_index=True)

    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


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

    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


def test_negative_series_behavior(non_positive_df: pd.DataFrame):
    """Check BoxCoxPreprocessing behavior in case of negative-value series"""
    preprocess = BoxCoxTransform()
    with pytest.raises(ValueError):
        _ = preprocess.fit_transform(df=non_positive_df)


@pytest.mark.parametrize(
    "preprocessing_class,method", ((BoxCoxTransform, "box-cox"), (YeoJohnsonTransform, "yeo-johnson"))
)
def test_transform_value(positive_df: pd.DataFrame, preprocessing_class: Any, method: str):
    """Check the value of transform result."""
    preprocess = preprocessing_class()
    value = preprocess.fit_transform(df=positive_df.copy())
    true_values = PowerTransformer(method=method).fit_transform(positive_df.values)
    npt.assert_array_almost_equal(value.values, true_values)


@pytest.mark.parametrize("preprocessing_class", (BoxCoxTransform, YeoJohnsonTransform))
def test_inverse_transform(positive_df: pd.DataFrame, preprocessing_class: Any):
    """Check that inverse_transform rolls back transform result."""
    preprocess = preprocessing_class()
    transformed_target = preprocess.fit_transform(df=positive_df.copy())
    inversed_target = preprocess.inverse_transform(df=transformed_target)
    np.testing.assert_array_almost_equal(inversed_target.values, positive_df.values)
