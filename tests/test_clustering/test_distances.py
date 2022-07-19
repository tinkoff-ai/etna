from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.clustering.distances.dtw_distance import DTWDistance
from etna.clustering.distances.dtw_distance import simple_dist
from etna.clustering.distances.euclidean_distance import EuclideanDistance
from etna.datasets import TSDataset


@pytest.fixture
def two_series() -> Tuple[pd.Series, pd.Series]:
    """Generate two series with different timestamp range."""
    x1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=10)})
    x1["target"] = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    x1.set_index("timestamp", inplace=True)

    x2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-02", periods=10)})
    x2["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x2.set_index("timestamp", inplace=True)

    return x1["target"], x2["target"]


@pytest.fixture
def pattern():
    x = [1] * 5 + [20, 3, 1, -5, -7, -8, -9, -10, -7.5, -6.5, -5, -4, -3, -2, -1, 0, 0, 1, 1] + [-1] * 11
    return x


@pytest.fixture
def dtw_ts(pattern) -> TSDataset:
    """Get df with complex pattern with timestamp lag."""
    dfs = []
    for i in range(1, 8):
        date_range = pd.date_range(f"2020-01-0{str(i)}", periods=35)
        tmp = pd.DataFrame({"timestamp": date_range})
        tmp["segment"] = str(i)
        tmp["target"] = pattern
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D")
    return ts


@pytest.mark.parametrize("trim_series,expected", ((True, 0), (False, 3)))
def test_euclidean_distance_no_trim_series(two_series: Tuple[pd.Series, pd.Series], trim_series: bool, expected: float):
    """Test euclidean distance in case of no trim series."""
    x1, x2 = two_series
    distance = EuclideanDistance(trim_series=trim_series)
    d = distance(x1, x2)
    assert d == expected


@pytest.mark.parametrize("trim_series,expected", ((True, 0), (False, 1)))
def test_dtw_distance_no_trim_series(two_series: Tuple[pd.Series, pd.Series], trim_series: bool, expected: float):
    """Test dtw distance in case of no trim series."""
    x1, x2 = two_series
    distance = DTWDistance(trim_series=trim_series)
    d = distance(x1, x2)
    assert d == expected


@pytest.mark.parametrize(
    "x1,x2,expected", (([1, 5, 4, 2], [1, 2, 4, 1], 3), ([1, 5, 4, 2], [1, 2, 4], 4), ([1, 5, 4], [1, 2, 4, 1], 5))
)
def test_dtw_different_length(x1: List[float], x2: List[float], expected: float):
    """Check dtw with different series' lengths."""
    x1 = pd.Series(x1)
    x2 = pd.Series(x2)
    dtw = DTWDistance()
    d = dtw(x1=x1, x2=x2)
    assert d == expected


@pytest.mark.parametrize(
    "x1,x2,expected",
    (
        (
            np.array([1, 8, 9, 2, 5]),
            np.array([4, 8, 7, 5]),
            np.array([[3, 10, 16, 20], [7, 3, 4, 7], [12, 4, 5, 8], [14, 10, 9, 8], [15, 13, 11, 8]]),
        ),
        (
            np.array([6, 3, 2, 1, 6]),
            np.array([3, 2, 1, 5, 8, 19, 0]),
            np.array(
                [
                    [3, 7, 12, 13, 15, 28, 34],
                    [3, 4, 6, 8, 13, 29, 31],
                    [4, 3, 4, 7, 13, 30, 31],
                    [6, 4, 3, 7, 14, 31, 31],
                    [9, 8, 8, 4, 6, 19, 25],
                ]
            ),
        ),
    ),
)
def test_dtw_build_matrix(x1: np.array, x2: np.array, expected: np.array):
    """Test dtw matrix computation."""
    dtw = DTWDistance()
    matrix = dtw._build_matrix(x1, x2, points_distance=simple_dist)
    np.testing.assert_array_equal(matrix, expected)


@pytest.mark.parametrize(
    "matrix,expected_path",
    (
        (
            np.array([[3, 10, 16, 20], [7, 3, 4, 7], [12, 4, 5, 8], [14, 10, 9, 8], [15, 13, 11, 8]]),
            [(4, 3), (3, 3), (2, 2), (1, 1), (0, 0)],
        ),
        (
            np.array(
                [
                    [3, 7, 12, 13, 15, 28, 34],
                    [3, 4, 6, 8, 13, 29, 31],
                    [4, 3, 4, 7, 13, 30, 31],
                    [6, 4, 3, 7, 14, 31, 31],
                    [9, 8, 8, 4, 6, 19, 25],
                ]
            ),
            [(4, 6), (4, 5), (4, 4), (4, 3), (3, 2), (2, 1), (1, 0), (0, 0)],
        ),
    ),
)
def test_path(matrix: np.array, expected_path: List[Tuple[int, int]]):
    """Check that DTWDistance reconstructs path correctly."""
    dtw = DTWDistance()
    path = dtw._get_path(matrix=matrix)
    assert len(path) == len(expected_path)
    for coords, expected_coords in zip(path, expected_path):
        assert coords == expected_coords


def test_dtw_get_average(dtw_ts: TSDataset):
    """Check that dtw centroid catches the pattern of df series."""
    dtw = DTWDistance()
    centroid = dtw.get_average(dtw_ts)
    percentiles = np.linspace(0, 1, 19)
    for segment in dtw_ts.segments:
        tmp = dtw_ts[:, segment, :][segment].dropna()
        for p in percentiles:
            assert abs(np.percentile(centroid["target"].values, p) - np.percentile(tmp["target"].values, p)) < 0.3
