from typing import List

import numpy as np
import pytest

from etna.analysis.outliers.density_outliers import get_anomalies_density
from etna.analysis.outliers.density_outliers import get_segment_density_outliers_indices
from etna.datasets.tsdataset import TSDataset


@pytest.fixture
def simple_window() -> np.array:
    return np.array([4, 5, 6, 4, 100, 200, 2])


@pytest.mark.parametrize(
    "window_size,n_neighbors,distance_threshold,expected",
    (
        (5, 2, 2.5, [4, 5, 6]),
        (6, 3, 10, [4, 5]),
        (2, 1, 1.8, [3, 4, 5, 6]),
        (3, 1, 120, []),
        (100, 2, 1.5, [2, 4, 5, 6]),
    ),
)
def test_get_segment_density_outliers_indices(
    simple_window: np.array, window_size: int, n_neighbors: int, distance_threshold: float, expected: List[int]
):
    """Check that outliers in one series computation works correctly."""
    outliers = get_segment_density_outliers_indices(
        series=simple_window, window_size=window_size, n_neighbors=n_neighbors, distance_threshold=distance_threshold
    )
    np.testing.assert_array_equal(outliers, expected)


def test_get_anomalies_density_interface(outliers_tsds: TSDataset):
    outliers = get_anomalies_density(ts=outliers_tsds, window_size=7, distance_coef=2, n_neighbors=3)
    for segment in ["1", "2"]:
        assert segment in outliers
        assert isinstance(outliers[segment], list)


def test_get_anomalies_density(outliers_tsds: TSDataset):
    """Check if get_anomalies_density works correctly."""
    outliers = get_anomalies_density(ts=outliers_tsds, window_size=7, distance_coef=2.1, n_neighbors=3)
    expected = {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])


def test_in_column(outliers_df_with_two_columns):
    outliers = get_anomalies_density(ts=outliers_df_with_two_columns, in_column="feature", window_size=10)
    expected = {"1": [np.datetime64("2021-01-08")], "2": [np.datetime64("2021-01-26")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])
