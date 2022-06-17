from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms.base import Transform


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    zero = "zero"
    mean = "mean"
    running_mean = "running_mean"
    forward_fill = "forward_fill"
    seasonal = "seasonal"


class TimeSeriesImputerTransform(Transform):
    """Transform to fill NaNs in series of a given dataframe.

    - It is assumed that given series begins with first non NaN value.

    - This transform can't fill NaNs in the future, only on train data.

    - This transform can't fill NaNs if all values are NaNs. In this case exception is raised.

    Warning
    -------
    This transform can suffer from look-ahead bias in 'mean' mode. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str = "target",
        strategy: str = ImputerMode.zero.value,
        window: int = -1,
        seasonality: int = 1,
        default_value: Optional[float] = None,
    ):
        """
        Create instance of TimeSeriesImputerTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        strategy:
            filling value in missing timestamps:

            - If "zero", then replace missing dates with zeros

            - If "mean", then replace missing dates using the mean in fit stage.

            - If "running_mean" then replace missing dates using mean of subset of data

            - If "forward_fill" then replace missing dates using last existing value

            - If "seasonal" then replace missing dates using seasonal moving average

        window:
            In case of moving average and seasonality.

            * If ``window=-1`` all previous dates are taken in account

            * Otherwise only window previous dates

        seasonality:
            the length of the seasonality
        default_value:
            value which will be used to impute the NaNs left after applying the imputer with the chosen strategy

        Raises
        ------
        ValueError:
            if incorrect strategy given
        """
        self.in_column = in_column
        self.strategy = strategy
        self.window = window
        self.seasonality = seasonality
        self.default_value = default_value
        self._strategy = ImputerMode(strategy)
        self._fill_value: Dict[str, int] = {}
        self._nan_timestamps: Dict[str, List[pd.Timestamp]] = {}

    def fit(self, df: pd.DataFrame) -> "TimeSeriesImputerTransform":
        """Fit params.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: TimeSeriesImputerTransform
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        features = df.loc[:, pd.IndexSlice[segments, self.in_column]]
        if features.isna().all().any():
            raise ValueError("Series hasn't non NaN values which means it is empty and can't be filled.")

        for segment in segments:
            series = features.loc[:, pd.IndexSlice[segment, self.in_column]]
            series = series[series.first_valid_index() :]
            self._nan_timestamps[segment] = series[series.isna()].index

        if self._strategy == ImputerMode.mean:
            mean_values = features.mean().to_dict()
            # take only segment from multiindex key
            mean_values = {key[0]: value for key, value in mean_values.items()}
            self._fill_value = mean_values

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nans in the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        segments = sorted(set(df.columns.get_level_values("segment")))

        cur_nans = {}
        for segment in segments:
            series = df.loc[:, pd.IndexSlice[segment, self.in_column]]
            cur_nans[segment] = series[series.isna()].index

        result_df = self._fill(df)

        # restore nans not in self.nan_timestamps
        for segment in segments:
            restore_nans = cur_nans[segment].difference(self._nan_timestamps[segment])
            result_df.loc[restore_nans, pd.IndexSlice[segment, self.in_column]] = np.nan

        return result_df

    def _fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new Series taking all previous dates and adding missing dates.

        Fills missed values for new dates according to ``self.strategy``

        Parameters
        ----------
        df: pd.DataFrame
            dataframe to fill

        Returns
        -------
        result: pd.DataFrame
        """
        if len(self._nan_timestamps) == 0:
            raise ValueError("Trying to apply the unfitted transform! First fit the transform.")

        segments = sorted(set(df.columns.get_level_values("segment")))
        result_df = df.copy(deep=True)

        if self._strategy == ImputerMode.zero:
            # we can't just do `result_df.fillna(value=0)`, it leads to errors if category dtype is present
            result_df.loc[:, pd.IndexSlice[segments, self.in_column]] = result_df.loc[
                :, pd.IndexSlice[segments, self.in_column]
            ].fillna(value=0)
        elif self._strategy == ImputerMode.forward_fill:
            result_df.fillna(method="ffill", inplace=True)
        elif self._strategy == ImputerMode.mean:
            for segment in segments:
                result_df.loc[:, pd.IndexSlice[segment, self.in_column]].fillna(
                    value=self._fill_value[segment], inplace=True
                )
        elif self._strategy == ImputerMode.running_mean or self._strategy == ImputerMode.seasonal:
            for segment in segments:
                history = self.seasonality * self.window if self.window != -1 else len(df)
                timestamps = list(df.index)
                for timestamp in self._nan_timestamps[segment]:
                    i = timestamps.index(timestamp)
                    indexes = np.arange(i - self.seasonality, i - self.seasonality - history, -self.seasonality)
                    indexes = indexes[indexes >= 0]
                    values = result_df.loc[result_df.index[indexes], pd.IndexSlice[segment, self.in_column]]
                    result_df.loc[timestamp, pd.IndexSlice[segment, self.in_column]] = np.nanmean(values)

        if self.default_value:
            result_df = result_df.fillna(value=self.default_value)
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.DataFrame
            transformed series
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        result_df = df.copy()

        for segment in segments:
            index = result_df.index.intersection(self._nan_timestamps[segment])
            result_df.loc[index, pd.IndexSlice[segment, self.in_column]] = np.nan
        return result_df


__all__ = ["TimeSeriesImputerTransform"]
