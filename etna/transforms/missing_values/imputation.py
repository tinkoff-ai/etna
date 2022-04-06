from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    zero = "zero"
    mean = "mean"
    running_mean = "running_mean"
    forward_fill = "forward_fill"


class _OneSegmentTimeSeriesImputerTransform(Transform):
    """One segment version of transform to fill NaNs in series of a given dataframe.

    - It is assumed that given series begins with first non NaN value.

    - This transform can't fill NaNs in the future, only on train data.

    - This transform can't fill NaNs if all values are NaNs. In this case exception is raised.

    """

    def __init__(self, in_column: str = "target", strategy: str = ImputerMode.zero, window: int = -1):
        """
        Create instance of _OneSegmentTimeSeriesImputerTransform.

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

        window:
            In case of moving average.

            * If ``window=-1`` all previous dates are taken in account

            * Otherwise only window previous dates

        Raises
        ------
        ValueError:
            if incorrect strategy given
        """
        self.in_column = in_column
        self.strategy = ImputerMode(strategy)
        self.window = window
        self.fill_value: Optional[int] = None
        self.nan_timestamps = None

    def fit(self, df: pd.DataFrame) -> "_OneSegmentTimeSeriesImputerTransform":
        """
        Fit preprocess params.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with series to fit preprocess params with

        Returns
        -------
        self: _OneSegmentTimeSeriesImputerTransform
            fitted preprocess
        """
        raw_series = df[self.in_column]
        if np.all(raw_series.isna()):
            raise ValueError("Series hasn't non NaN values which means it is empty and can't be filled.")
        series = raw_series[raw_series.first_valid_index() :]
        self.nan_timestamps = series[series.isna()].index
        if self.strategy == ImputerMode.zero:
            self.fill_value = 0
        elif self.strategy == ImputerMode.mean:
            self.fill_value = series.mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform given series.

        Parameters
        ----------
        df: pd.Dataframe
            transform ``in_column`` series of given dataframe

        Returns
        -------
        result: pd.DataFrame
            dataframe with in_column series with filled gaps
        """
        result_df = df.copy()
        cur_nans = result_df[result_df[self.in_column].isna()].index

        result_df[self.in_column] = self._fill(result_df[self.in_column])

        # restore nans not in self.nan_timestamps
        restore_nans = cur_nans.difference(self.nan_timestamps)
        result_df.loc[restore_nans, self.in_column] = np.nan

        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform dataframe.

        Parameters
        ----------
        df: pd.Dataframe
            inverse transform ``in_column`` series of given dataframe

        Returns
        -------
        result: pd.DataFrame
            dataframe with in_column series with initial values
        """
        result_df = df.copy()
        index = result_df.index.intersection(self.nan_timestamps)
        result_df.loc[index, self.in_column] = np.nan
        return result_df

    def _fill(self, df: pd.Series) -> pd.Series:
        """
        Create new Series taking all previous dates and adding missing dates.

        Fills missed values for new dates according to ``self.strategy``

        Parameters
        ----------
        df: pd.Series
            series to fill

        Returns
        -------
        result: pd.Series
        """
        if self.fill_value is not None:
            df = df.fillna(value=self.fill_value)
        elif self.strategy == ImputerMode.forward_fill:
            df = df.fillna(method="ffill")
        elif self.strategy == ImputerMode.running_mean:
            for i, val in enumerate(df):
                if pd.isnull(val):
                    left_bound = max(i - self.window, 0) if self.window != -1 else 0
                    df.iloc[i] = df.iloc[left_bound:i].mean()
        return df


class TimeSeriesImputerTransform(PerSegmentWrapper):
    """Transform to fill NaNs in series of a given dataframe.

    - It is assumed that given series begins with first non NaN value.

    - This transform can't fill NaNs in the future, only on train data.

    - This transform can't fill NaNs if all values are NaNs. In this case exception is raised.

    Warning
    -------
    This transform can suffer from look-ahead bias in 'mean' mode. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(self, in_column: str = "target", strategy: str = ImputerMode.zero, window: int = -1):
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

        window:
            In case of moving average.

            * If ``window=-1`` all previous dates are taken in account

            * Otherwise only window previous dates

        Raises
        ------
        ValueError:
            if incorrect strategy given
        """
        self.in_column = in_column
        self.strategy = strategy
        self.window = window
        super().__init__(transform=_OneSegmentTimeSeriesImputerTransform(self.in_column, self.strategy, self.window))


__all__ = ["TimeSeriesImputerTransform"]
