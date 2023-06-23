from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

import numpy as np
import pandas as pd

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import check_new_segments


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    mean = "mean"
    running_mean = "running_mean"
    forward_fill = "forward_fill"
    seasonal = "seasonal"
    constant = "constant"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Supported strategies: {', '.join([repr(m.value) for m in cls])}"
        )


class TimeSeriesImputerTransform(ReversibleTransform):
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
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.strategy = strategy
        self.window = window
        self.seasonality = seasonality
        self.default_value = default_value
        self.constant_value = constant_value
        self._strategy = ImputerMode(strategy)
        self._fill_value: Optional[Dict[str, float]] = None
        self._nan_timestamps: Optional[Dict[str, List[pd.Timestamp]]] = None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def _fit(self, df: pd.DataFrame):
        """Fit the transform.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        features = df.loc[:, pd.IndexSlice[segments, self.in_column]]
        if features.isna().all().any():
            raise ValueError("Series hasn't non NaN values which means it is empty and can't be filled.")

        nan_timestamps = {}
        for segment in segments:
            series = features.loc[:, pd.IndexSlice[segment, self.in_column]]
            series = series[series.first_valid_index() :]
            nan_timestamps[segment] = series[series.isna()].index

        fill_value = {}
        if self._strategy is ImputerMode.mean:
            mean_values = features.mean().to_dict()
            # take only segment from multiindex key
            mean_values = {key[0]: value for key, value in mean_values.items()}
            fill_value = mean_values

        self._nan_timestamps = nan_timestamps
        self._fill_value = fill_value

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.

        Returns
        -------
        :
            Transformed Dataframe in etna wide format.
        """
        if self._fill_value is None or self._nan_timestamps is None:
            raise ValueError("Transform is not fitted!")

        segments = sorted(set(df.columns.get_level_values("segment")))
        check_new_segments(transform_segments=segments, fit_segments=self._nan_timestamps.keys())

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
        """Fill the NaNs in a given Dataframe.

        Fills missed values for new dates according to ``self.strategy``

        Parameters
        ----------
        df:
            dataframe to fill

        Returns
        -------
        :
            Filled Dataframe.
        """
        self._fill_value = cast(Dict[str, float], self._fill_value)
        self._nan_timestamps = cast(Dict[str, List[pd.Timestamp]], self._nan_timestamps)
        segments = sorted(set(df.columns.get_level_values("segment")))

        if self._strategy is ImputerMode.constant:
            new_values = df.loc[:, pd.IndexSlice[:, self.in_column]].fillna(value=self.constant_value)
            df.loc[:, pd.IndexSlice[:, self.in_column]] = new_values
        elif self._strategy is ImputerMode.forward_fill:
            new_values = df.loc[:, pd.IndexSlice[:, self.in_column]].fillna(method="ffill")
            df.loc[:, pd.IndexSlice[:, self.in_column]] = new_values
        elif self._strategy is ImputerMode.mean:
            for segment in segments:
                df.loc[:, pd.IndexSlice[segment, self.in_column]].fillna(value=self._fill_value[segment], inplace=True)
        elif self._strategy is ImputerMode.running_mean or self._strategy is ImputerMode.seasonal:
            timestamp_to_index = {timestamp: i for i, timestamp in enumerate(df.index)}
            for segment in segments:
                history = self.seasonality * self.window if self.window != -1 else len(df)
                for timestamp in self._nan_timestamps[segment]:
                    i = timestamp_to_index[timestamp]
                    indexes = np.arange(i - self.seasonality, i - self.seasonality - history, -self.seasonality)
                    indexes = indexes[indexes >= 0]
                    values = df.loc[df.index[indexes], pd.IndexSlice[segment, self.in_column]]
                    df.loc[timestamp, pd.IndexSlice[segment, self.in_column]] = np.nanmean(values)

        if self.default_value is not None:
            df.fillna(value=self.default_value, inplace=True)
        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform dataframe.

        Parameters
        ----------
        df:
            Dataframe to be inverse transformed.

        Returns
        -------
        :
            Dataframe after applying inverse transformation.
        """
        if self._fill_value is None or self._nan_timestamps is None:
            raise ValueError("Transform is not fitted!")

        segments = sorted(set(df.columns.get_level_values("segment")))
        check_new_segments(transform_segments=segments, fit_segments=self._nan_timestamps.keys())

        for segment in segments:
            index = df.index.intersection(self._nan_timestamps[segment])
            df.loc[index, pd.IndexSlice[segment, self.in_column]] = np.NaN
        return df

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``strategy``, ``window``.
        Other parameters are expected to be set by the user.

        Strategy "seasonal" is suggested only if ``self.seasonality`` is set higher than 1.

        Returns
        -------
        :
            Grid to tune.
        """
        if self.seasonality > 1:
            return {
                "strategy": CategoricalDistribution(["constant", "mean", "running_mean", "forward_fill", "seasonal"]),
                "window": IntDistribution(low=1, high=20),
            }
        else:
            return {
                "strategy": CategoricalDistribution(["constant", "mean", "running_mean", "forward_fill"]),
                "window": IntDistribution(low=1, high=20),
            }


__all__ = ["TimeSeriesImputerTransform"]
