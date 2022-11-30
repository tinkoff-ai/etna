from typing import Any

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.math import MADTransform
from etna.transforms.math import MaxTransform
from etna.transforms.math import MeanTransform
from etna.transforms.math import MedianTransform
from etna.transforms.math import MinMaxDifferenceTransform
from etna.transforms.math import MinTransform
from etna.transforms.math import QuantileTransform
from etna.transforms.math import StdTransform
from etna.transforms.math import SumTransform


@pytest.fixture
def simple_df_for_agg() -> pd.DataFrame:
    n = 10
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n)})
    df["target"] = list(range(n))
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_for_agg() -> pd.DataFrame:
    n = 10
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n)})
    df["target"] = [-1, 1, 3, 2, 4, 9, 8, 5, 6, 0]
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def df_for_agg_with_nan() -> pd.DataFrame:
    n = 10
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n)})
    df["target"] = [-1, 1, 3, None, 4, 9, 8, 5, 6, 0]
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df)
    return df


@pytest.mark.parametrize(
    "class_name,out_column",
    (
        (MaxTransform, None),
        (MaxTransform, "test_max"),
        (MinTransform, None),
        (MinTransform, "test_min"),
        (MedianTransform, None),
        (MedianTransform, "test_median"),
        (MeanTransform, None),
        (MeanTransform, "test_mean"),
        (StdTransform, None),
        (StdTransform, "test_std"),
        (MADTransform, None),
        (MADTransform, "test_mad"),
        (MinMaxDifferenceTransform, None),
        (MinMaxDifferenceTransform, "test_min_max_diff"),
        (SumTransform, None),
        (SumTransform, "test_sum"),
    ),
)
def test_interface_simple(simple_df_for_agg: pd.DataFrame, class_name: Any, out_column: str):
    transform = class_name(window=3, out_column=out_column, in_column="target")
    res = transform.fit_transform(df=simple_df_for_agg)
    result_column = out_column if out_column is not None else transform.__repr__()
    assert sorted(res["segment_1"]) == sorted([result_column] + ["target"])


@pytest.mark.parametrize("out_column", (None, "test_q"))
def test_interface_quantile(simple_df_for_agg: pd.DataFrame, out_column: str):
    transform = QuantileTransform(quantile=0.7, window=4, out_column=out_column, in_column="target")
    res = transform.fit_transform(df=simple_df_for_agg)
    result_column = out_column if out_column is not None else transform.__repr__()
    assert sorted(res["segment_1"]) == sorted([result_column] + ["target"])


@pytest.mark.parametrize(
    "window,seasonality,alpha,periods,fill_na,expected",
    (
        (10, 1, 1, 1, 0, np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])),
        (-1, 1, 1, 1, 0, np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])),
        (3, 1, 1, 1, -17, np.array([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8])),
        (3, 1, 0.5, 1, -17, np.array([0, 0.5, 2.5 / 3, 4.25 / 3, 2, 7.75 / 3, 9.5 / 3, 11.25 / 3, 13 / 3, 14.75 / 3])),
        (
            3,
            1,
            0.5,
            3,
            -12,
            np.array([-12, -12, 2.5 / 3, 4.25 / 3, 2, 7.75 / 3, 9.5 / 3, 11.25 / 3, 13 / 3, 14.75 / 3]),
        ),
        (3, 2, 1, 1, -17, np.array([0, 1, 1, 2, 2, 3, 4, 5, 6, 7])),
    ),
)
def test_mean_feature(
    simple_df_for_agg: pd.DataFrame,
    window: int,
    seasonality: int,
    alpha: float,
    periods: int,
    fill_na: float,
    expected: np.array,
):
    transform = MeanTransform(
        window=window,
        seasonality=seasonality,
        alpha=alpha,
        min_periods=periods,
        fillna=fill_na,
        in_column="target",
        out_column="result",
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["result"]).all()


@pytest.mark.parametrize(
    "window,seasonality,periods,fill_na,expected",
    (
        (10, 1, 1, 0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (-1, 1, 1, 0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (3, 1, 1, -17, np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])),
        (3, 2, 1, -17, np.array([0, 1, 0, 1, 0, 1, 2, 3, 4, 5])),
    ),
)
def test_min_feature(
    simple_df_for_agg: pd.DataFrame, window: int, seasonality: int, periods: int, fill_na: float, expected: np.array
):
    transform = MinTransform(
        window=window,
        seasonality=seasonality,
        min_periods=periods,
        fillna=fill_na,
        in_column="target",
        out_column="result",
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["result"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (10, 1, 0, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        (-1, 1, 0, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        (3, 2, -17, np.array([-17, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
    ),
)
def test_max_feature(simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array):
    transform = MaxTransform(
        window=window, min_periods=periods, fillna=fill_na, in_column="target", out_column="result"
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["result"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (3, 3, -17, np.array([-17, -17, 1, 2, 3, 4, 5, 6, 7, 8])),
        (-1, 1, -17, np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])),
    ),
)
def test_median_feature(simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array):
    transform = MedianTransform(
        window=window, min_periods=periods, fillna=fill_na, in_column="target", out_column="result"
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["result"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (
            3,
            3,
            -17,
            np.array(
                [
                    -17,
                    -17,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            ),
        ),
        (
            3,
            1,
            -17,
            np.array(
                [
                    -17,
                    (1 / 2) ** 0.5,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            ),
        ),
    ),
)
def test_std_feature(simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array):
    transform = StdTransform(
        window=window, min_periods=periods, fillna=fill_na, in_column="target", out_column="result"
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["result"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (3, 3, -17, [-17, -17, 4 / 3, 2 / 3, 2 / 3, 8 / 3, 2, 14 / 9, 10 / 9, 22 / 9]),
        (4, 1, -17, [0, 1, 4 / 3, 1.25, 1, 2.25, 2.75, 2, 1.5, 9.5 / 4]),
        (-1, 1, 0, [0, 1, 4 / 3, 1.25, 1.44, 7 / 3, 138 / 49, 2.625, 208 / 81, 27 / 10]),
    ),
)
def test_mad_transform(df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.ndarray):
    transform = MADTransform(
        window=window, min_periods=periods, fillna=fill_na, in_column="target", out_column="result"
    )
    res = transform.fit_transform(df_for_agg)
    np.testing.assert_array_almost_equal(expected, res["segment_1"]["result"])


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    ((3, 3, -17, [-17, -17, 4 / 3, -17, -17, -17, 2, 14 / 9, 10 / 9, 22 / 9]),),
)
def test_mad_transform_with_nans(
    df_for_agg_with_nan: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.ndarray
):
    transform = MADTransform(
        window=window, min_periods=periods, fillna=fill_na, in_column="target", out_column="result"
    )
    res = transform.fit_transform(df_for_agg_with_nan)
    np.testing.assert_array_almost_equal(expected, res["segment_1"]["result"])


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (10, 1, 0, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        (-1, 1, 0, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        (3, 2, -17, np.array([-17, 1, 2, 2, 2, 2, 2, 2, 2, 2])),
    ),
)
def test_min_max_diff_feature(
    simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array
):
    transform = MinMaxDifferenceTransform(
        window=window, min_periods=periods, fillna=fill_na, in_column="target", out_column="result"
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["result"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    ((10, 1, 0, np.array([-1, 0, 3, 3, 7, 16, 24, 29, 35, 35])),),
)
def test_sum_feature_with_nan(
    df_for_agg_with_nan: pd.DataFrame,
    window: int,
    periods: int,
    fill_na: float,
    expected: np.ndarray,
):
    transform = SumTransform(
        window=window,
        min_periods=periods,
        fillna=fill_na,
        in_column="target",
        out_column="result",
    )
    res = transform.fit_transform(df_for_agg_with_nan)
    np.testing.assert_array_almost_equal(expected, res["segment_1"]["result"])


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (10, 1, 0, np.array([0, 1, 3, 6, 10, 15, 21, 28, 36, 45])),
        (-1, 1, 0, np.array([0, 1, 3, 6, 10, 15, 21, 28, 36, 45])),
        (3, 1, -17, np.array([0, 1, 3, 6, 9, 12, 15, 18, 21, 24])),
        (1, 1, -17, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        (3, 3, -17, np.array([-17, -17, 3, 6, 9, 12, 15, 18, 21, 24])),
    ),
)
def test_sum_feature(
    simple_df_for_agg: pd.DataFrame,
    window: int,
    periods: int,
    fill_na: float,
    expected: np.array,
):
    transform = SumTransform(
        window=window,
        min_periods=periods,
        fillna=fill_na,
        in_column="target",
        out_column="result",
    )

    res = transform.fit_transform(simple_df_for_agg)
    np.testing.assert_array_almost_equal(expected, res["segment_1"]["result"])


@pytest.mark.parametrize(
    "transform",
    (
        MaxTransform(in_column="target", window=5),
        MinTransform(in_column="target", window=5),
        MedianTransform(in_column="target", window=5),
        MeanTransform(in_column="target", window=5),
        StdTransform(in_column="target", window=5),
        MADTransform(in_column="target", window=5),
        MinMaxDifferenceTransform(in_column="target", window=5),
        SumTransform(in_column="target", window=5),
    ),
)
def test_fit_transform_with_nans(transform, ts_diff_endings):
    ts_diff_endings.fit_transform([transform])
