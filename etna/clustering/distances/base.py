import sys
import warnings
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd

from etna.core import BaseMixin

if TYPE_CHECKING:
    from etna.datasets import TSDataset


class Distance(ABC, BaseMixin):
    """Base class for distances between series."""

    def __init__(self, trim_series: bool = False, inf_value: float = sys.float_info.max // 10**200):
        """Init Distance.

        Parameters
        ----------
        trim_series:

            * if True, get common (according to timestamp index) part of series and compute distance with it;

            * if False, compute distance with given series without any modifications.

        inf_value:
            if two empty series given or series' indices interception is empty,
            return ``inf_value`` as a distance between the series
        """
        self.trim_series = trim_series
        self.inf_value = inf_value

    @abstractmethod
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between two given arrays."""
        pass

    def __call__(self, x1: pd.Series, x2: pd.Series) -> float:
        """Compute distance between x1 and x2.

        Parameters
        ----------
        x1:
            timestamp-indexed series
        x2:
            timestamp-indexed series

        Returns
        -------
        float:
            distance between x1 and x2
        """
        if self.trim_series:
            common_indices = x1.index.intersection(x2.index)
            _x1, _x2 = x1[common_indices], x2[common_indices]
        else:
            _x1, _x2 = x1, x2

        # TODO: better to avoid such comments
        # if x1 and x2 have no interception with timestamp return inf_value as a distance
        if _x1.empty and _x2.empty:
            return self.inf_value

        distance = self._compute_distance(x1=_x1.values, x2=_x2.values)
        # TODO: better to avoid such comments
        # use it to avoid clustering confusing: if the last if passes we need to clip all the distances
        # to inf_value
        distance = min(self.inf_value, distance)
        return distance

    @staticmethod
    def _validate_dataset(ts: "TSDataset"):
        """Check that dataset does not contain NaNs."""
        for segment in ts.segments:
            series = ts[:, segment, "target"]
            first_valid_index = 0
            last_valid_index = series.reset_index(drop=True).last_valid_index()
            series_length = last_valid_index - first_valid_index + 1
            if len(series.dropna()) != series_length:
                warnings.warn(
                    f"Timeseries contains NaN values, which will be dropped. "
                    f"If it is not desirable behaviour, handle them manually."
                )
                break

    @abstractmethod
    def _get_average(self, ts: "TSDataset") -> pd.DataFrame:
        """Get series that minimizes squared distance to given ones according to the Distance."""
        pass

    def get_average(self, ts: "TSDataset", **kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Get series that minimizes squared distance to given ones according to the Distance.

        Parameters
        ----------
        ts:
            TSDataset with series to be averaged
        kwargs:
            additional parameters for averaging

        Returns
        -------
        pd.DataFrame:
            dataframe with columns "timestamp" and "target" that contains the series
        """
        self._validate_dataset(ts)
        centroid = self._get_average(ts, **kwargs)  # type: ignore
        return centroid


__all__ = ["Distance"]
