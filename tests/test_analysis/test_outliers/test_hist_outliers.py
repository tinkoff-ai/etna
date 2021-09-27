from typing import List

import numpy as np
import pytest

from etna.analysis.outliers.hist_outliers import computeF
from etna.analysis.outliers.hist_outliers import hist
from etna.analysis.outliers.hist_outliers import v_optimal_hist
from etna.datasets.tsdataset import TSDataset


@pytest.mark.parametrize(
    "series,B,expected",
    (
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 1, 60),
        (np.array([1, 2, 3, 4, -1, 0, -2, -2, -1]), 2, 7.8),
        (np.array([1, 2, 3, 100, 36, 64, -1, 0, -2, -2, -1]), 4, 396.8),
        (np.array([1, 2, 3, 4, 5, 6, 6, 7]), 7, 0),
    ),
)
def test_v_optimal_hist(series: np.array, B: int, expected: float):
    """Check that v_optimal_hist works correctly."""
    error = v_optimal_hist(series, B)
    assert error == expected


@pytest.mark.parametrize(
    "series,k", ((np.random.random(100), 10), (np.random.random(100), 20), (np.random.random(10), 4))
)
def test_computeF_format(series: np.array, k: int):
    """Check that computeF produce the correct size output."""
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2
    _, idx = computeF(series, k, p, pp)
    for ai in range(len(series)):
        for bi in range(ai + 1, len(series)):
            for ci in range(1, min(bi - ai + 1, k + 1)):
                for i in range(len(idx[ai][bi][ci])):
                    assert len(idx[ai][bi][ci][i]) == ci


@pytest.mark.parametrize(
    "series,k,dim,expected",
    (
        (
            np.array([1, 0, 2, 3, 5]),
            3,
            0,
            np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [2, 0.5, 0, 0], [5, 2, 0.5, 0], [14.8, 5, 2, 0.5]]),
        ),
        (
            np.array([-6, -3, 0, -6, -1]),
            3,
            0,
            np.array([[0, 0, 0, 0], [4.5, 0, 0, 0], [18, 4.5, 0, 0], [24.75, 6, 0, 0], [30.8, 18, 6, 0]]),
        ),
        (
            np.array([1, 2, 3, 1, 5, 2]),
            3,
            2,
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [8, 2, 0, 0], [8.75, 2, 0.5, 0]]),
        ),
    ),
)
def test_computeF(series: np.array, k: int, dim: int, expected: np.array):
    """Check that computeF works correctly."""
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2
    res, idx = computeF(series, k, p, pp)
    np.testing.assert_almost_equal(res[dim][4], expected[4])


@pytest.mark.parametrize(
    "series,B,expected",
    (
        (np.array([1, 0, 1, -1, 0, 4, 1, 0, 1, 0, 1, 1, 0, 0, -1, 0, 0]), 5, np.array([3, 5, 14])),
        (np.arange(40), 5, np.array([])),
        (np.array([4, 5, 4, 3, 9, 10, 8, 2, 1, 0, 1, 1, 5, 1, 2]), 4, np.array([12])),
    ),
)
def test_hist(series: np.array, B: int, expected: np.array):
    """Check that hist works correctly."""
    anomal = hist(series, B)
    np.testing.assert_array_equal(anomal, expected)
