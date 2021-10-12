from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.analysis import get_sequence_anomalies
from etna.datasets import TSDataset
from etna.transforms.base import Transform


class OutliersTransform(Transform, ABC):
    """Finds outliers in specific columns of DataFrame and replaces it with NaNs."""

    def __init__(self, in_column: str):
        """
        Create instance of OutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        """
        self.in_column = in_column
        self.outliers_timestamps: Optional[Dict[str, List[pd.Timestamp]]] = None
        self.original_values: Optional[Dict[str, List[pd.Timestamp]]] = None

    def _save_original_values(self, ts: TSDataset):
        """
        Save values to be replaced with NaNs.

        Parameters
        ----------
        ts:
            original TSDataset
        """
        self.original_values = dict()

        for segment, timestamps in self.outliers_timestamps.items():
            segment_ts = ts[:, segment, :]
            segment_values = segment_ts[segment_ts.index.isin(timestamps)].droplevel("segment", axis=1)[self.in_column]
            self.original_values[segment] = segment_values

    def fit(self, df: pd.DataFrame) -> "OutliersTransform":
        """
        Find outliers using detection method.

        Parameters
        ----------
        df:
            dataframe with series to find outliers

        Returns
        -------
        result: OutliersTransform
            instance with saved outliers
        """
        ts = TSDataset(df, freq=pd.infer_freq(df.index))
        self.outliers_timestamps = self.detect_outliers(ts)
        self._save_original_values(ts)

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
        for segment in df.columns.get_level_values("segment").unique():
            result_df.loc[self.outliers_timestamps[segment], pd.IndexSlice[segment, self.in_column]] = np.NaN
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation. Returns back deleted values.

        Parameters
        ----------
        df:
            data to transform

        Returns
        -------
        result: pd.DataFrame
            data with reconstructed values
        """
        result = df.copy()

        for segment in self.original_values.keys():
            segment_ts = result[segment, self.in_column]
            segment_ts[segment_ts.index.isin(self.outliers_timestamps[segment])] = self.original_values[segment]

        return result

    @abstractmethod
    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call function for detection outliers with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        dict of outliers:
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        pass


class MedianOutliersTransform(OutliersTransform):
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
        super().__init__(in_column=self.in_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call `get_anomalies_median` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        dict of outliers:
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_median(ts, self.window_size, self.alpha)


class DensityOutliersTransform(OutliersTransform):
    """Transform that uses get_anomalies_density to find anomalies in data."""

    def __init__(
        self,
        in_column: str,
        window_size: int = 15,
        distance_coef: float = 3,
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
        distance_coef:
            factor for standard deviation that forms distance threshold to determine points are close to each other
        n_neighbors:
            min number of close neighbors of point not to be outlier
        distance_func:
            distance function
        """
        self.in_column = in_column
        self.window_size = window_size
        self.distance_coef = distance_coef
        self.n_neighbors = n_neighbors
        self.distance_func = distance_func
        super().__init__(in_column=self.in_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call `get_anomalies_density` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        dict of outliers:
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_density(ts, self.window_size, self.distance_coef, self.n_neighbors, self.distance_func)


class SAXOutliersTransform(OutliersTransform):
    """Transform that uses get_sequence_anomalies to find anomalies in data and replaces them with NaN."""

    def __init__(
        self,
        in_column: str,
        num_anomalies: int = 1,
        anomaly_lenght: int = 15,
        alphabet_size: int = 3,
        word_lenght: int = 3,
    ):
        """Create instance of SAXOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        num_anomalies:
            number of outliers to be found
        anomaly_lenght:
            target lenght of outliers
        alphabet_size:
            the number of letters with which the subsequence will be encrypted
        word_lenght:
            the number of segments into which the subsequence will be divided by the paa algorithm
        """
        self.in_column = in_column
        self.num_anomalies = num_anomalies
        self.anomaly_lenght = anomaly_lenght
        self.alphabet_size = alphabet_size
        self.word_lenght = word_lenght
        super().__init__(in_column=self.in_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call `get_sequence_anomalies` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process
        Returns
        -------
        dict of outliers:
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_sequence_anomalies(
            ts=ts,
            num_anomalies=self.num_anomalies,
            anomaly_lenght=self.anomaly_lenght,
            alphabet_size=self.alphabet_size,
            word_lenght=self.word_lenght,
            in_column=self.in_column,
        )


__all__ = ["MedianOutliersTransform", "DensityOutliersTransform"]
