from typing import Callable
from typing import List
from typing import Tuple

import numba
import numpy as np
import pandas as pd

from etna.clustering.distances.base import Distance


@numba.cfunc(numba.float64(numba.float64, numba.float64))
def simple_dist(x1: float, x2: float) -> float:
    """Get distance between two samples for dtw distance.

    Parameters
    ----------
    x1: float
        first value
    x2: float
        second value

    Returns
    -------
    distance: float
        distance between x1 and x2
    """
    return abs(x1 - x2)


class DTWDistance(Distance):
    """DTW distance handler."""

    def __init__(self, points_distance: Callable[[np.array, np.array], float] = simple_dist, trim_series: bool = False):
        """Init DTWDistance.

        Parameters
        ----------
        points_distance: callable
            function to be used for computation of distance between two series' points
        trim_series: bool
            True if it is necessary to trim series, default False.
        """
        super().__init__(trim_series=trim_series)
        self.points_distance = points_distance

    @staticmethod
    @numba.njit
    def _build_matrix(x1: np.array, x2: np.array, points_distance: Callable[[float, float], float]) -> np.array:
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
    def _get_path(matrix: np.array) -> List[Tuple[int, int]]:
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

    def _compute_distance(self, x1: np.array, x2: np.array) -> float:
        """Compute distance between x1 and x2."""
        matrix = self._build_matrix(x1=x1, x2=x2, points_distance=self.points_distance)
        return matrix[-1][-1]

    def _dba_iteration(self, initial_centroid: np.array, series_list: List[np.array]) -> np.array:
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

    def get_average(self, xs: pd.DataFrame, n_iters: int = 10) -> pd.DataFrame:
        """Get series that minimizes squared distance to given ones according to the dtw distance.

        Parameters
        ----------
        xs: pd.DataFrame
            dataframe with columns "segment", "timestamp", "target" that contains series to be averaged

        Returns
        -------
        centroid: pd.DataFrame
            dataframe with columns "timestamp" and "target" that contains the series
        """
        # Let the longest series be the first initialisation of centroid
        segment_length_df = xs.groupby(["segment"])["target"].count()
        biggest_segment_idx = segment_length_df.argmax()
        biggest_segment = segment_length_df.reset_index()["segment"].loc[biggest_segment_idx]
        initial_centroid_df = xs[xs["segment"] == biggest_segment]
        centroid = initial_centroid_df["target"].values

        series_list = []
        segments = xs["segment"].unique()
        for segment in segments:
            series_list.append(xs[xs["segment"] == segment]["target"].values)

        # Repeat _dba_iteration n_iters time to adjust centroid with series
        for _ in range(n_iters):
            new_centroid = self._dba_iteration(initial_centroid=centroid, series_list=series_list)
            centroid = new_centroid
        centroid_df = pd.DataFrame({"timestamp": initial_centroid_df["timestamp"].values, "target": centroid})
        return centroid_df


__all__ = ["DTWDistance", "simple_dist"]
