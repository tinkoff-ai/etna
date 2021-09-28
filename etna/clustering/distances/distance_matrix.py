from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from etna.clustering.distances.base import Distance


class DistanceMatrix:
    """DistanceMatrix computes distance matrix from dataframe in default ETNA format."""

    def __init__(self, distance: Distance):
        """Init DistanceMatrix.

        Parameters
        ----------
        distance: Distance
            class for distance measurement
        """
        self.distance = distance
        self.matrix: Optional[np.array] = None
        self.series: Optional[List[np.array]] = None
        self.segment2idx: Dict[str, int] = {}
        self.idx2segment: Dict[int, str] = {}
        self.series_number: Optional[int] = None

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame):
        """Check that dataframe contains only the main columns."""
        if not sorted(df.columns) == ["segment", "target", "timestamp"]:
            raise ValueError("error")

    def _get_series(self, df: pd.DataFrame) -> List[np.array]:
        """Parse given dataframe and get timestamp-indexed segment series.
        Build mapping from segment to idx in matrix and vice versa.
        """
        series = []
        df.set_index("timestamp", inplace=True)
        segments = sorted(df["segment"].unique())
        for i, segment in enumerate(segments):
            self.segment2idx[segment] = i
            self.idx2segment[i] = segment
            tmp_series = df[df["segment"] == segment]["target"]
            series.append(tmp_series)
        df.reset_index(inplace=True)
        self.series_number = len(segments)
        return series

    def _compute_dist(self, series: List[pd.Series], idx: int) -> np.array:
        """Compute distance from idx-th series to other ones."""
        distances = np.array([self.distance(series[idx], series[j]) for j in range(self.series_number)])
        return distances

    def _compute_dist_matrix(self, series: List[np.array]) -> np.array:
        """Compute distance matrix for given series."""
        distances = np.empty(shape=(self.series_number, self.series_number))
        for idx in tqdm(range(self.series_number)):
            distances[idx] = self._compute_dist(series=series, idx=idx)
        return distances

    def fit(self, df: pd.DataFrame) -> "DistanceMatrix":
        """Fit distance matrix: get timeseries from df and compute pairwise distances.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with columns "segment", "timestamp", "target"

        Returns
        -------
        self:
            fitted DistanceMatrix object

        Raises
        ------
        ValueError:
            if there are any extra columns in df or any required ones are absent
        """
        self._validate_dataframe(df=df)
        df.sort_values(["segment", "timestamp"], inplace=True)
        self.series = self._get_series(df=df)
        self.matrix = self._compute_dist_matrix(series=self.series)
        return self

    def predict(self) -> np.array:
        """Get distance matrix.

        Returns
        -------
        matrix: np.array
            2D array with distances between series
        """
        return self.matrix

    def fit_predict(self, df: pd.DataFrame) -> np.array:
        """Compute distance matrix and return it.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with time series to compute matrix with

        Returns
        -------
        matrix: np.array
            2D array with distances between series
        """
        return self.fit(df=df).predict()


__all__ = ["DistanceMatrix"]
