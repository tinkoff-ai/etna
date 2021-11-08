from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd

from etna.clustering.distances.base import Distance

if TYPE_CHECKING:
    from etna.datasets import TSDataset


@numba.cfunc(numba.float64(numba.float64[:], numba.float64[:]))
def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Get euclidean distance between two arrays.

    Parameters
    ----------
    x1:
        first array
    x2:
        second array

    Returns
    -------
    float:
        distance between x1 and x2
    """
    return np.linalg.norm(x1 - x2)


class EuclideanDistance(Distance):
    """Euclidean distance handler."""

    def __init__(self, trim_series: bool = True):
        """Init EuclideanDistance.

        Parameters
        ----------
        trim_series:
            if True, compare parts of series with common timestamp
        """
        super().__init__(trim_series=trim_series)

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between x1 and x2."""
        return euclidean_distance(x1=x1, x2=x2)

    def _get_average(self, ts: "TSDataset") -> pd.DataFrame:
        """Get series that minimizes squared distance to given ones according to the euclidean distance.

        Parameters
        ----------
        ts:
            TSDataset with series to be averaged

        Returns
        -------
        pd.DataFrame:
            dataframe with columns "timestamp" and "target" that contains the series
        """
        centroid = pd.DataFrame({"timestamp": ts.index.values, "target": ts.df.mean(axis=1).values})
        return centroid


__all__ = ["EuclideanDistance", "euclidean_distance"]
