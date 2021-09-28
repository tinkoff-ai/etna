import sys
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd


class Distance(ABC):
    """Base class for distances between series."""

    def __init__(self, trim_series: bool = False, inf_value: float = sys.float_info.max // 10 ** 200):
        """Init Distance.

        Parameters
        ----------
        trim_series: bool
            if True, get common (according to timestamp index) part of series and compute distance with it; if False,
            compute distance with given series without any modifications.
        inf_value: float
            if two empty series given or series' indices interception is empty,
            return inf_value as a distance between the series
        """
        self.trim_series = trim_series
        self.inf_value = inf_value

    @abstractmethod
    def _compute_distance(self, x1: np.array, x2: np.array) -> float:
        """Compute distance between two given arrays."""
        pass

    def __call__(self, x1: pd.Series, x2: pd.Series) -> float:
        """Compute distance between x1 and x2.

        Parameters
        ----------
        x1: pd.Series
            timestamp-indexed series
        x2: pd.Series
            timestamp-indexed series

        Returns
        -------
        distance: float
            distance between x1 and x2
        """
        if self.trim_series:
            indices = x1.index.intersection(x2.index)
            _x1, _x2 = x1[indices], x2[indices]
        else:
            _x1, _x2 = x1, x2

        # if x1 and x2 have no interception with timestamp return inf_value as a distance
        if _x1.empty and _x2.empty:
            return self.inf_value

        distance = self._compute_distance(x1=_x1.values, x2=_x2.values)
        # use it to avoid clustering confusing: if the last if passes we need to clip all the distances
        # to inf_value
        distance = min(self.inf_value, distance)
        return distance

    def get_average(self, xs: pd.DataFrame) -> pd.DataFrame:
        """Get series that minimizes squared distance to given ones according to the Distance.

        Parameters
        ----------
        xs: pd.DataFrame
            dataframe with columns "segment", "timestamp", "target" that contains series to be averaged

        Returns
        -------
        centroid: pd.DataFrame
            dataframe with columns "timestamp" and "target" that contains the series
        """
        centroid = xs.groupby("timestamp")[["target"]].mean()
        centroid.reset_index(inplace=True)
        return centroid


__all__ = ["Distance"]
