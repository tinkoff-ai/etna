import warnings
from enum import Enum
from typing import List
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
    seasonal = "seasonal"
    constant = "constant"


class _OneSegmentTimeSeriesImputerTransform(Transform):
    """One segment version of transform to fill NaNs in series of a given dataframe.

    - It is assumed that given series begins with first non NaN value.

    - This transform can't fill NaNs in the future, only on train data.

    - This transform can't fill NaNs if all values are NaNs. In this case exception is raised.

    """

    def __init__(
        self,
        in_column: str,
        strategy: str,
        window: int,
        seasonality: int,
        default_value: Optional[float],
        constant_value: float = 0,
    ):
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

            - If "seasonal" then replace missing dates using seasonal moving average

            - If "constant" then replace missing dates using constant value.

        window:
            In case of moving average and seasonality.

            * If ``window=-1`` all previous dates are taken in account

            * Otherwise only window previous dates

        seasonality:
            the length of the seasonality
        default_value:
            value which will be used to impute the NaNs left after applying the imputer with the chosen strategy
        constant_value:
            value to fill gaps in "constant" strategy

        Raises
        ------
        ValueError:
            if incorrect strategy given
        """
        self.in_column = in_column
        self.strategy = ImputerMode(strategy)
        self.window = window
        self.seasonality = seasonality
        self.default_value = default_value
        self.constant_value = constant_value
        self.fill_value: Optional[float] = None
        self.nan_timestamps: Optional[List[pd.Timestamp]] = None

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
            warnings.warn(
                "zero strategy will be removed in etna 2.0.0. Use constant strategy instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.fill_value = 0
        if self.strategy == ImputerMode.constant:
            self.fill_value = self.constant_value
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
        if self.nan_timestamps is None:
            raise ValueError("Trying to apply the unfitted transform! First fit the transform.")

        if (
            self.strategy == ImputerMode.zero
            or self.strategy == ImputerMode.mean
            or self.strategy == ImputerMode.constant
        ):
            df = df.fillna(value=self.fill_value)
        elif self.strategy == ImputerMode.forward_fill:
            df = df.fillna(method="ffill")
        elif self.strategy == ImputerMode.running_mean or self.strategy == ImputerMode.seasonal:
            history = self.seasonality * self.window if self.window != -1 else len(df)
            timestamps = list(df.index)
            for timestamp in self.nan_timestamps:
                i = timestamps.index(timestamp)
                indexes = np.arange(i - self.seasonality, i - self.seasonality - history, -self.seasonality)
                indexes = indexes[indexes >= 0]
                df.iloc[i] = np.nanmean(df.iloc[indexes])

        if self.default_value:
            df = df.fillna(value=self.default_value)
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

    def __init__(
        self,
        in_column: str = "target",
        strategy: str = ImputerMode.constant,
        window: int = -1,
        seasonality: int = 1,
        default_value: Optional[float] = None,
        constant_value: float = 0,
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

            - If "constant" then replace missing dates using constant value.

        window:
            In case of moving average and seasonality.

            * If ``window=-1`` all previous dates are taken in account

            * Otherwise only window previous dates

        seasonality:
            the length of the seasonality
        default_value:
            value which will be used to impute the NaNs left after applying the imputer with the chosen strategy
        constant_value:
            value to fill gaps in "constant" strategy

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
        self.constant_value = constant_value
        super().__init__(
            transform=_OneSegmentTimeSeriesImputerTransform(
                in_column=self.in_column,
                strategy=self.strategy,
                window=self.window,
                seasonality=self.seasonality,
                default_value=self.default_value,
                constant_value=self.constant_value,
            )
        )


__all__ = ["TimeSeriesImputerTransform"]
