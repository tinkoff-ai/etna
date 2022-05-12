from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import Transform


class TimeFlagsTransform(Transform, FutureMixin):
    """TimeFlagsTransform is a class that implements extraction of the main time-based features from datetime column."""

    def __init__(
        self,
        minute_in_hour_number: bool = True,
        fifteen_minutes_in_hour_number: bool = False,
        hour_number: bool = True,
        half_hour_number: bool = False,
        half_day_number: bool = False,
        one_third_day_number: bool = False,
        out_column: Optional[str] = None,
    ):
        """Initialise class attributes.

        Parameters
        ----------
        minute_in_hour_number:
            if True: add column with minute number to feature dataframe in transform
        fifteen_minutes_in_hour_number:
            if True: add column with number of fifteen-minute interval within hour with numeration from 0
            to feature dataframe in transform
        hour_number:
            if True: add column with hour number to feature dataframe in transform
        half_hour_number:
            if True: add column with 0 for the first half of the hour and 1 for the second
            to feature dataframe in transform
        half_day_number:
            if True: add column with 0 for the first half of the day and 1 for the second
            to feature dataframe in transform
        one_third_day_number:
            if True: add column with number of 8-hour interval within day with numeration from 0
            to feature dataframe in transform
        out_column:
            base for the name of created columns;

            * if set the final name is '{out_column}_{feature_name}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        Raises
        ------
        ValueError: if feature has invalid initial params
        """
        if not any(
            [
                minute_in_hour_number,
                fifteen_minutes_in_hour_number,
                hour_number,
                half_hour_number,
                half_day_number,
                one_third_day_number,
            ]
        ):
            raise ValueError(
                f"{type(self).__name__} feature does nothing with given init args configuration, "
                f"at least one of minute_in_hour_number, fifteen_minutes_in_hour_number, hour_number, "
                f"half_hour_number, half_day_number, one_third_day_number should be True."
            )

        self.date_column_name = None
        self.minute_in_hour_number: bool = minute_in_hour_number
        self.fifteen_minutes_in_hour_number: bool = fifteen_minutes_in_hour_number
        self.hour_number: bool = hour_number
        self.half_hour_number: bool = half_hour_number
        self.half_day_number: bool = half_day_number
        self.one_third_day_number: bool = one_third_day_number

        self.out_column = out_column

        # create empty init parameters
        self._empty_parameters = dict(
            minute_in_hour_number=False,
            fifteen_minutes_in_hour_number=False,
            hour_number=False,
            half_hour_number=False,
            half_day_number=False,
            one_third_day_number=False,
        )

    def _get_column_name(self, feature_name: str) -> str:
        if self.out_column is None:
            init_parameters = deepcopy(self._empty_parameters)
            init_parameters[feature_name] = self.__dict__[feature_name]
            temp_transform = TimeFlagsTransform(**init_parameters, out_column=self.out_column)  # type: ignore
            return repr(temp_transform)
        else:
            return f"{self.out_column}_{feature_name}"

    def fit(self, *args, **kwargs) -> "TimeFlagsTransform":
        """Fit datetime model."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method for features based on time.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: pd.DataFrame
            Dataframe with extracted features
        """
        features = pd.DataFrame(index=df.index)
        timestamp_series = pd.Series(df.index)

        if self.minute_in_hour_number:
            minute_in_hour_number = self._get_minute_number(timestamp_series=timestamp_series)
            features[self._get_column_name("minute_in_hour_number")] = minute_in_hour_number

        if self.fifteen_minutes_in_hour_number:
            fifteen_minutes_in_hour_number = self._get_period_in_hour(
                timestamp_series=timestamp_series, period_in_minutes=15
            )
            features[self._get_column_name("fifteen_minutes_in_hour_number")] = fifteen_minutes_in_hour_number

        if self.hour_number:
            hour_number = self._get_hour_number(timestamp_series=timestamp_series)
            features[self._get_column_name("hour_number")] = hour_number

        if self.half_hour_number:
            half_hour_number = self._get_period_in_hour(timestamp_series=timestamp_series, period_in_minutes=30)
            features[self._get_column_name("half_hour_number")] = half_hour_number

        if self.half_day_number:
            half_day_number = self._get_period_in_day(timestamp_series=timestamp_series, period_in_hours=12)
            features[self._get_column_name("half_day_number")] = half_day_number

        if self.one_third_day_number:
            one_third_day_number = self._get_period_in_day(timestamp_series=timestamp_series, period_in_hours=8)
            features[self._get_column_name("one_third_day_number")] = one_third_day_number

        for feature in features.columns:
            features[feature] = features[feature].astype("category")

        dataframes = []
        for seg in df.columns.get_level_values("segment").unique():
            tmp = df[seg].join(features)
            _idx = tmp.columns.to_frame()
            _idx.insert(0, "segment", seg)
            tmp.columns = pd.MultiIndex.from_frame(_idx)
            dataframes.append(tmp)

        result = pd.concat(dataframes, axis=1).sort_index(axis=1)
        result.columns.names = ["segment", "feature"]
        return result

    @staticmethod
    def _get_minute_number(timestamp_series: pd.Series) -> np.ndarray:
        """Generate array with the minute number in the hour."""
        return timestamp_series.apply(lambda x: x.minute).values

    @staticmethod
    def _get_period_in_hour(timestamp_series: pd.Series, period_in_minutes: int = 15) -> np.ndarray:
        """Generate an array with the period number in the hour.

        Accepts a period length in minutes as input and returns array where timestamps marked by period number.
        """
        return timestamp_series.apply(lambda x: x.minute // period_in_minutes).values

    @staticmethod
    def _get_hour_number(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the hour number in the day."""
        return timestamp_series.apply(lambda x: x.hour).values

    @staticmethod
    def _get_period_in_day(timestamp_series: pd.Series, period_in_hours: int = 12) -> np.ndarray:
        """Generate an array with the period number in the day.

        Accepts a period length in hours as input and returns array where timestamps marked by period number.
        """
        return timestamp_series.apply(lambda x: x.hour // period_in_hours).values


__all__ = ["TimeFlagsTransform"]
