import warnings
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from etna.clustering.distances.base import Distance
from etna.core import BaseMixin


class DistanceMatrix(BaseMixin):
    """DistanceMatrix computes distance matrix from TSDataset."""

    def __init__(self, distance: Distance):
        """Init DistanceMatrix.

        Parameters
        ----------
        distance:
            class for distance measurement
        """
        self.distance = distance
        self.matrix: Optional[np.array] = None
        self.series: Optional[List[np.array]] = None
        self.segment2idx: Dict[str, int] = {}
        self.idx2segment: Dict[int, str] = {}
        self.series_number: Optional[int] = None

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

    def _get_series(self, ts: "TSDataset") -> List[pd.Series]:
        """Parse given TSDataset and get timestamp-indexed segment series.
        Build mapping from segment to idx in matrix and vice versa.
        """
        series_list = []
        for i, segment in enumerate(ts.segments):
            self.segment2idx[segment] = i
            self.idx2segment[i] = segment
            series = ts[:, segment, "target"].dropna()
            series_list.append(series)

        self.series_number = len(series_list)
        return series_list

    def _compute_dist(self, series: List[pd.Series], idx: int) -> np.array:
        """Compute distance from idx-th series to other ones."""
        distances = np.array([self.distance(series[idx], series[j]) for j in range(self.series_number)])
        return distances

    def _compute_dist_matrix(self, series: List[pd.Series]) -> np.array:
        """Compute distance matrix for given series."""
        distances = np.empty(shape=(self.series_number, self.series_number))
        for idx in tqdm(range(self.series_number)):
            distances[idx] = self._compute_dist(series=series, idx=idx)
        return distances

    def fit(self, ts: "TSDataset") -> "DistanceMatrix":
        """Fit distance matrix: get timeseries from ts and compute pairwise distances.

        Parameters
        ----------
        ts:
            TSDataset with timeseries

        Returns
        -------
        self:
            fitted DistanceMatrix object

        """
        self._validate_dataset(ts)
        self.series = self._get_series(ts)
        self.matrix = self._compute_dist_matrix(self.series)
        return self

    def predict(self) -> np.array:
        """Get distance matrix.

        Returns
        -------
        matrix:
            2D array with distances between series
        """
        return self.matrix

    def fit_predict(self, ts: "TSDataset") -> np.array:
        """Compute distance matrix and return it.

        Parameters
        ----------
        ts:
           TSDataset with timeseries to compute matrix with

        Returns
        -------
        matrix:
            2D array with distances between series
        """
        return self.fit(ts).predict()


__all__ = ["DistanceMatrix"]
