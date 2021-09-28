import numba
import numpy as np

from etna.clustering.distances.base import Distance


@numba.cfunc(numba.float64(numba.float64[:], numba.float64[:]))
def euclidean_distance(x1: np.array, x2: np.array) -> float:
    """Get euclidean distance between two arrays.

    Parameters
    ----------
    x1: float
        first array
    x2: float
        second array

    Returns
    -------
    distance: float
        distance between x1 and x2
    """
    return np.linalg.norm(x1 - x2)


class EuclideanDistance(Distance):
    """Euclidean distance handler."""

    def __init__(self, trim_series: bool = True):
        """Init EuclideanDistance.

        Parameters
        ----------
        trim_series: bool
            if True, compare parts of series with common timestamp
        """
        super().__init__(trim_series=trim_series)

    def _compute_distance(self, x1: np.array, x2: np.array) -> float:
        return euclidean_distance(x1=x1, x2=x2)


__all__ = ["EuclideanDistance", "euclidean_distance"]
