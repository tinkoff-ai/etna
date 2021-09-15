from enum import Enum

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
    """Fills Nans in series of DataFrame with zeros, previous value, average or moving average."""

    def __init__(self, in_column: str = "target", strategy: str = ImputerMode.zero, window: int = -1):
        """
        Create instance of _OneSegmentTimeSeriesImputerTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        strategy:
            filling value in missed dates:
            - If "zero", then replace missing dates with zeros
            - If "mean", then replace missing dates using the mean in fit stage.
            - If "running_mean" then replace missing dates using mean of subset of data
            - If "forward_fill" then replace missing dates using last existing value
        window:
            In case of moving average.
            If window=-1 all previous dates are taken in account
            Otherwise only window previous dates

        Raises
        ------
        ValueError:
            if incorrect strategy given
        """
        self.in_column = in_column
        self.strategy = ImputerMode(strategy)
        self.window = window
        self.fill_value = None

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
        if self.strategy == ImputerMode.zero:
            self.fill_value = 0
        elif self.strategy == ImputerMode.mean:
            self.fill_value = df[self.in_column].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform given series.

        Parameters
        ----------
        df: pd.Dataframe
            transform in_column series of given dataframe

        Returns
        -------
        result: pd.DataFrame
            dataframe with in_column series with filled gaps
        """
        result_df = df.copy()
        result_df[self.in_column] = self._fill(df[self.in_column])
        return result_df

    def _fill(self, df: pd.Series) -> pd.Series:
        """
        Create new Series taking all previous dates and adding missing dates.

        Fills missed values for new dates according to filling_type

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
            df = df.fillna(value=0)
        elif self.strategy == ImputerMode.running_mean:
            for i, val in enumerate(df):
                if pd.isnull(val):
                    left_bound = max(i - self.window, 0) if self.window != -1 else 0
                    df.iloc[i] = df.iloc[left_bound:i].mean()
        return df


class TimeSeriesImputerTransform(PerSegmentWrapper):
    """TimeSeriesImputerTransform fills the gaps in series from given dataframe."""

    def __init__(self, in_column: str = "target", strategy: str = ImputerMode.zero, window: int = -1):
        """
        Create instance of TimeSeriesImputerTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        strategy:
            filling value in missed dates:
            - If "zero", then replace missing dates with zeros
            - If "mean", then replace missing dates using the mean in fit stage.
            - If "running_mean" then replace missing dates using mean of subset of data
            - If "forward_fill" then replace missing dates using last existing value
        window:
            In case of moving average.
            If window=-1 all previous dates are taken in account
            Otherwise only window previous dates

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
