from typing import TYPE_CHECKING
from typing import Callable
from typing import List
from typing import Tuple

import numba
import numpy as np
import pandas as pd

from etna.clustering.distances.base import Distance

if TYPE_CHECKING:
    from etna.datasets import TSDataset


@numba.njit
def simple_dist(x1: float, x2: float) -> float:
    """Get distance between two samples for dtw distance.

    Parameters
    ----------
    x1:
        first value
    x2:
        second value

    Returns
    -------
    float:
        distance between x1 and x2
    """
    return abs(x1 - x2)


class DTWDistance(Distance):
    """DTW distance handler."""

    def __init__(self, points_distance: Callable[[float, float], float] = simple_dist, trim_series: bool = False):
        """Init DTWDistance.

        Parameters
        ----------
        points_distance:
            function to be used for computation of distance between two series' points
        trim_series:
            True if it is necessary to trim series, default False.

        Notes
        -----
        Specifying manual ``points_distance`` might slow down the clustering algorithm.
        """
        super().__init__(trim_series=trim_series)
        self.points_distance = points_distance

    @staticmethod
    @numba.njit
    def _build_matrix(x1: np.ndarray, x2: np.ndarray, points_distance: Callable[[float, float], float]) -> np.ndarray:
        """Build dtw-distance matrix for series x1 and x2."""
        x1_size, x2_size = len(x1), len(x2)
        matrix = np.empty(shape=(x1_size, x2_size))
        matrix[0][0] = points_distance(x1[0], x2[0])
        for i in range(1, x1_size):
            matrix[i][0] = points_distance(x1[i], x2[0]) + matrix[i - 1][0]
        for j in range(1, x2_size):
            matrix[0][j] = points_distance(x1[0], x2[j]) + matrix[0][j - 1]
        for i in range(1, x1_size):
            for j in range(1, x2_size):
                matrix[i][j] = points_distance(x1[i], x2[j]) + min(
                    matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]
                )
        return matrix

    @staticmethod
    @numba.njit
    def _get_path(matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Build a warping path with given matrix of dtw-distance."""
        i, j = matrix.shape[0] - 1, matrix.shape[1] - 1
        path = [(i, j)]
        while i and j:
            candidates = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]
            costs = np.array([matrix[c] for c in candidates])
            k = np.argmin(costs)
            i, j = candidates[k]
            path.append((i, j))
        while i:
            i = i - 1
            path.append((i, j))
        while j:
            j = j - 1
            path.append((i, j))
        return path

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between x1 and x2."""
        matrix = self._build_matrix(x1=x1, x2=x2, points_distance=self.points_distance)
        return matrix[-1][-1]

    def _dba_iteration(self, initial_centroid: np.ndarray, series_list: List[np.ndarray]) -> np.ndarray:
        """Run DBA iteration.
        * for each series from series list build a dtw matrix and warping path
        * update values of centroid with values from series according to path
        """
        assoc_table = initial_centroid.copy()
        n_samples = np.ones(shape=(len(initial_centroid)))
        for series in series_list:
            mat = self._build_matrix(x1=initial_centroid, x2=series, points_distance=self.points_distance)
            path = self._get_path(matrix=mat)
            i, j = len(initial_centroid) - 1, len(series) - 1
            while i and j:
                assoc_table[i] += series[j]
                n_samples[i] += 1
                path.pop(0)
                i, j = path[0]
        centroid = assoc_table / n_samples
        return centroid

    @staticmethod
    def _get_longest_series(ts: "TSDataset") -> pd.Series:
        """Get the longest series from the list."""
        series_list: List[pd.Series] = []
        for segment in ts.segments:
            series = ts[:, segment, "target"].dropna()
            series_list.append(series)
        longest_series = max(series_list, key=len)
        return longest_series

    @staticmethod
    def _get_all_series(ts: "TSDataset") -> List[np.ndarray]:
        """Get series from the TSDataset."""
        series_list = []
        for segment in ts.segments:
            series = ts[:, segment, "target"].dropna().values
            series_list.append(series)
        return series_list

    def _get_average(self, ts: "TSDataset", n_iters: int = 10) -> pd.DataFrame:
        """Get series that minimizes squared distance to given ones according to the dtw distance.

        Parameters
        ----------
        ts:
            TSDataset with series to be averaged
        n_iters:
            number of DBA iterations to adjust centroid with series

        Returns
        -------
        pd.Dataframe:
            dataframe with columns "timestamp" and "target" that contains the series
        """
        series_list = self._get_all_series(ts)
        initial_centroid = self._get_longest_series(ts)
        centroid = initial_centroid.values
        for _ in range(n_iters):
            new_centroid = self._dba_iteration(initial_centroid=centroid, series_list=series_list)
            centroid = new_centroid
        centroid = pd.DataFrame({"timestamp": initial_centroid.index.values, "target": centroid})
        return centroid


__all__ = ["DTWDistance", "simple_dist"]
