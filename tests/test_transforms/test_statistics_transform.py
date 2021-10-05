from typing import Any
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.transforms.statistics import MaxTransform
from etna.transforms.statistics import MeanTransform
from etna.transforms.statistics import MedianTransform
from etna.transforms.statistics import MinTransform
from etna.transforms.statistics import QuantileTransform
from etna.transforms.statistics import StdTransform


@pytest.fixture
def simple_df_for_agg() -> pd.DataFrame:
    n = 10
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=n)})
    df["target"] = list(range(n))
    df["segment"] = "segment_1"

    df = df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


@pytest.mark.parametrize(
    "class_name,out_postfix,columns",
    (
        (MaxTransform, None, ["target_max"]),
        (MaxTransform, "test_max", ["target_test_max"]),
        (MinTransform, None, ["target_min"]),
        (MinTransform, "test_min", ["target_test_min"]),
        (MedianTransform, None, ["target_median"]),
        (MedianTransform, "test_median", ["target_test_median"]),
        (MeanTransform, None, ["target_mean"]),
        (MeanTransform, "test_mean", ["target_test_mean"]),
        (StdTransform, None, ["target_std"]),
        (StdTransform, "test_std", ["target_test_std"]),
    ),
)
def test_interface_simple(simple_df_for_agg: pd.DataFrame, class_name: Any, out_postfix: str, columns: List[str]):
    transform = class_name(window=3, out_postfix=out_postfix, in_column="target")
    res = transform.fit_transform(df=simple_df_for_agg)
    assert sorted(res["segment_1"]) == sorted(columns + ["target"])


@pytest.mark.parametrize(
    "quantile,out_postfix,columns",
    ((0.7, None, ["target_quantile_0.7"]), (0.9, "test_quantile", ["target_test_quantile_0.9"])),
)
def test_interface_quantile(simple_df_for_agg: pd.DataFrame, quantile: float, out_postfix: str, columns: List[str]):
    transform = QuantileTransform(quantile=quantile, window=4, out_postfix=out_postfix, in_column="target")
    res = transform.fit_transform(df=simple_df_for_agg)
    assert sorted(res["segment_1"]) == sorted(columns + ["target"])


@pytest.mark.parametrize(
    "window,seasonality,alpha,periods,fill_na,expected",
    (
        (10, 1, 1, 1, 0, np.array([0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])),
        (-1, 1, 1, 1, 0, np.array([0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])),
        (3, 1, 1, 1, -17, np.array([-17, 0, 0.5, 1, 2, 3, 4, 5, 6, 7])),
        (3, 1, 0.5, 1, -17, np.array([-17, 0, 0.5, 2.5 / 3, 4.25 / 3, 2, 7.75 / 3, 9.5 / 3, 11.25 / 3, 13 / 3])),
        (3, 1, 0.5, 3, -12, np.array([-12, -12, -12, 2.5 / 3, 4.25 / 3, 2, 7.75 / 3, 9.5 / 3, 11.25 / 3, 13 / 3])),
        (3, 2, 1, 1, -17, np.array([-17, 0, 1, 1, 2, 2, 3, 4, 5, 6])),
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
        window=window, seasonality=seasonality, alpha=alpha, min_periods=periods, fillna=fill_na, in_column="target"
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["target_mean"]).all()


@pytest.mark.parametrize(
    "window,seasonality,periods,fill_na,expected",
    (
        (10, 1, 1, 0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (-1, 1, 1, 0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (3, 1, 1, -17, np.array([-17, 0, 0, 0, 1, 2, 3, 4, 5, 6])),
        (3, 2, 1, -17, np.array([-17, 0, 1, 0, 1, 0, 1, 2, 3, 4])),
    ),
)
def test_min_feature(
    simple_df_for_agg: pd.DataFrame, window: int, seasonality: int, periods: int, fill_na: float, expected: np.array
):
    transform = MinTransform(
        window=window, seasonality=seasonality, min_periods=periods, fillna=fill_na, in_column="target"
    )
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["target_min"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (10, 1, 0, np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])),
        (-1, 1, 0, np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])),
        (3, 2, -17, np.array([-17, -17, 1, 2, 3, 4, 5, 6, 7, 8])),
    ),
)
def test_max_feature(simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array):
    transform = MaxTransform(window=window, min_periods=periods, fillna=fill_na, in_column="target")
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["target_max"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (3, 3, -17, np.array([-17, -17, -17, 1, 2, 3, 4, 5, 6, 7])),
        (-1, 1, -17, np.array([-17, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])),
    ),
)
def test_median_feature(simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array):
    transform = MedianTransform(window=window, min_periods=periods, fillna=fill_na, in_column="target")
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["target_median"]).all()


@pytest.mark.parametrize(
    "window,periods,fill_na,expected",
    (
        (3, 3, -17, np.array([-17, -17, -17, 1, 1, 1, 1, 1, 1, 1])),
        (3, 1, -17, np.array([-17, -17, np.sqrt(0.5 ** 2 * 2), 1, 1, 1, 1, 1, 1, 1])),
    ),
)
def test_std_feature(simple_df_for_agg: pd.DataFrame, window: int, periods: int, fill_na: float, expected: np.array):
    transform = StdTransform(window=window, min_periods=periods, fillna=fill_na, in_column="target")
    res = transform.fit_transform(simple_df_for_agg)
    res["expected"] = expected
    assert (res["expected"] == res["segment_1"]["target_std"]).all()
