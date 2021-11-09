import numpy as np
import pytest

from etna.analysis.outliers import get_anomalies_hist
from etna.analysis.outliers.hist_outliers import compute_f
from etna.analysis.outliers.hist_outliers import hist
from etna.analysis.outliers.hist_outliers import v_optimal_hist


@pytest.mark.parametrize(
    "series,bins_number,expected",
    (
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 1, 60),
        (np.array([1, 2, 3, 4, -1, 0, -2, -2, -1]), 2, 7.8),
        (np.array([1, 2, 3, 100, 36, 64, -1, 0, -2, -2, -1]), 4, 396.8),
        (np.array([1, 2, 3, 4, 5, 6, 6, 7]), 7, 0),
    ),
)
def test_v_optimal_hist_one_value(series: np.array, bins_number: int, expected: float):
    """Check that v_optimal_hist works correctly."""
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2
    error = v_optimal_hist(series, bins_number, p, pp)[len(series) - 1][bins_number - 1]
    assert error == expected


@pytest.mark.parametrize(
    "series,bins_number,expected",
    (
        (np.array([-1, 0, 4, 3, 8]), 2, np.array([[0, 0], [0.5, 0], [14, 0.5], [17, 1], [50.8, 14.5]])),
        (
            np.array([4, 2, 3, 5, 3, 1]),
            3,
            np.array([[0, 0, 0], [2, 0, 0], [2, 0.5, 0], [5, 2, 0.5], [5.2, 4, 2], [10, 5.2, 4]]),
        ),
    ),
)
def test_v_optimal_hist(series: np.array, bins_number: int, expected: np.array):
    """Check that v_optimal_hist works correctly."""
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2
    error = v_optimal_hist(series, bins_number, p, pp)
    np.testing.assert_almost_equal(error, expected)


@pytest.mark.parametrize("series_len,k", ((100, 10), (100, 20), (10, 4)))
def test_compute_f_format(random_seed, series_len: int, k: int):
    """Check that computeF produce the correct size output."""
    series = np.random.random(size=series_len)
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2
    _, idx = compute_f(series, k, p, pp)
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
        (
            np.array([1, 2, 3, 1, 5, 6]),
            3,
            2,
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [8, 2, 0, 0], [14.75, 4.66666667, 0.5, 0]]
            ),
        ),
    ),
)
def test_computef(series: np.array, k: int, dim: int, expected: np.array):
    """Check that computeF works correctly."""
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2
    res, _ = compute_f(series, k, p, pp)
    np.testing.assert_almost_equal(res[dim], expected)


@pytest.mark.parametrize(
    "series,bins_number,expected",
    (
        (np.array([1, 0, 1, -1, 0, 4, 1, 0, 1, 0, 1, 1, 0, 0, -1, 0, 0]), 5, np.array([3, 5, 14])),
        (np.array([4, 5, 4, 3, 9, 10, 8, 2, 1, 0, 1, 1, 5, 1, 2]), 4, np.array([12])),
    ),
)
def test_hist(series: np.array, bins_number: int, expected: np.array):
    """Check that hist works correctly."""
    anomalies = hist(series, bins_number)
    np.testing.assert_array_equal(anomalies, expected)


def test_in_column(outliers_df_with_two_columns):
    outliers = get_anomalies_hist(ts=outliers_df_with_two_columns, in_column="feature")
    expected = {"1": [np.datetime64("2021-01-08")], "2": [np.datetime64("2021-01-26")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])
