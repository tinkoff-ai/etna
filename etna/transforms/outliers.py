from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.datasets import TSDataset
from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentOutliersTransform(Transform):
    """Fills Nans in series of DataFrame with zeros, previous value, average or moving average."""

    def __init__(self, in_column: str, detection_method: Callable):
        """
        Create instance of _OneSegmentOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        detection_method:
            method to find outliers, should take TSDataset
            and return dict of outliers in format {segment: [outliers_timestamps]}
        """
        self.in_column = in_column
        self.detection_method = detection_method
        self.outliers_timestamps = None

    def fit(self, df: pd.DataFrame) -> "_OneSegmentOutliersTransform":
        """
        Find outliers using detection method.

        Parameters
        ----------
        df:
            dataframe with series to find outliers

        Returns
        -------
        result: _OneSegmentTimeSeriesImputerTransform
            instance with saved outliers
        """
        ts = TSDataset(df, freq=pd.infer_freq(df.index))
        outliers = self.detection_method(ts)
        key = outliers.keys()[0]
        self.outliers_timestamps = outliers[key]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace found outliers with NaNs.

        Parameters
        ----------
        df:
            transform in_column series of given dataframe

        Returns
        -------
        result: pd.DataFrame
            dataframe with in_column series with filled with NaNs
        """
        result_df = df.copy()
        result_df[self.outliers_timestamps, self.in_column] = np.NaN
        return result_df


class MedianOutliersTransform(PerSegmentWrapper):
    """Transform that uses get_anomalies_median to find anomalies in data."""

    def __init__(self, in_column: str, window_size: int = 10, alpha: float = 3):
        """Create instance of MedianOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        window_size:
            number of points in the window
        alpha:
            coefficient for determining the threshold
        """
        self.in_column = in_column
        self.window_size = window_size
        self.alpha = alpha
        super().__init__(
            transform=_OneSegmentOutliersTransform(in_column=self.in_column, detection_method=self._detect_outliers)
        )

    def _detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call `get_anomalies_median` function with self parameters."""
        return get_anomalies_median(ts, self.window_size, self.alpha)


class DensityOutliersTransform(PerSegmentWrapper):
    """Transform that uses get_anomalies_density to find anomalies in data."""

    def __init__(
        self,
        in_column: str,
        window_size: int = 15,
        distance_threshold: float = 100,
        n_neighbors: int = 3,
        distance_func: Callable[[float, float], float] = lambda x, y: abs(x - y),
    ):
        """Create instance of DensityOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        window_size:
            size of windows to build
        distance_threshold:
            distance threshold to determine points are close to each other
        n_neighbors:
            min number of close neighbors of point not to be outlier
        distance_func:
            distance function
        """
        self.in_column = in_column
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.n_neighbors = n_neighbors
        self.distance_func = distance_func
        super().__init__(
            transform=_OneSegmentOutliersTransform(in_column=self.in_column, detection_method=self._detect_outliers)
        )

    def _detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call `get_anomalies_density` function with self parameters."""
        return get_anomalies_density(
            ts, self.window_size, self.distance_threshold, self.n_neighbors, self.distance_func
        )


__all__ = ["MedianOutliersTransform", "DensityOutliersTransform"]
