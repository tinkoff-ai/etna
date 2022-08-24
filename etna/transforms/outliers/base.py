from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.transforms.base import ReversibleTransform


class OutliersTransform(ReversibleTransform, ABC):
    """Finds outliers in specific columns of DataFrame and replaces it with NaNs."""

    def __init__(self, in_column: str):
        """
        Create instance of OutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        """
        super().__init__(in_column=in_column)
        self.outliers_timestamps: Optional[Dict[str, List[pd.Timestamp]]] = None
        self.original_values: Optional[Dict[str, List[pd.Timestamp]]] = None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.

        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return []

    def _save_original_values(self, ts: TSDataset):
        """
        Save values to be replaced with NaNs.

        Parameters
        ----------
        ts:
            original TSDataset
        """
        if self.outliers_timestamps is None:
            raise ValueError("Something went wrong during outliers detection stage! Check the transform parameters.")
        self.original_values = dict()
        for segment, timestamps in self.outliers_timestamps.items():
            segment_ts = ts[:, segment, :]
            segment_values = segment_ts[segment_ts.index.isin(timestamps)].droplevel("segment", axis=1)[self.in_column]
            self.original_values[segment] = segment_values

    def _fit(self, df: pd.DataFrame) -> "OutliersTransform":
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

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace found outliers with NaNs.

        Parameters
        ----------
        df:
            transform ``in_column`` series of given dataframe

        Returns
        -------
        result: pd.DataFrame
            dataframe with in_column series with filled with NaNs
        """
        if self.outliers_timestamps is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
        for segment in df.columns.get_level_values("segment").unique():
            df.loc[self.outliers_timestamps[segment], pd.IndexSlice[segment, self.in_column]] = np.NaN
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if self.original_values is None or self.outliers_timestamps is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling inverse_transform method.")
        for segment in self.original_values.keys():
            segment_ts = df[segment, self.in_column]
            segment_ts[segment_ts.index.isin(self.outliers_timestamps[segment])] = self.original_values[segment]
        return df

    @abstractmethod
    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call function for detection outliers with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        pass
