from math import ceil
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd

from etna.transforms.base import Transform


class DateFlagsTransform(Transform):
    """DateFlagsTransform is a class that implements extraction of the main date-based features from datetime column.

    Notes
    -----
    tables with freq types in pandas: https://nagornyy.me/courses/data-science/dates-and-time-in-python-and-pandas
    """

    def __init__(
        self,
        day_number_in_week: Optional[bool] = True,
        day_number_in_month: Optional[bool] = True,
        week_number_in_month: Optional[bool] = False,
        week_number_in_year: Optional[bool] = False,
        month_number_in_year: Optional[bool] = False,
        year_number: Optional[bool] = False,
        special_days_in_week: Sequence[int] = (),
        special_days_in_month: Sequence[int] = (),
    ):
        """Create instance of DateFlags.

        Parameters
        ----------
        day_number_in_week:
            if True, add column "day_number_in_week" with weekday info to feature dataframe in transform
        day_number_in_month:
            if True, add column "day_number_in_month" with day info to feature dataframe in transform
        week_number_in_month:
            if True, add column "week_number_in_month" with week number (in month context) to feature dataframe
            in transform
        week_number_in_year:
            if True, add column "week_number_in_year" with week number (in year context) to feature dataframe
            in transform
        month_number_in_year:
            if True, add column "month_number_in_year" with month info to feature dataframe in transform
        year_number:
            if True, add column "year_number" with year info to feature dataframe in transform
        special_days_in_week:
            list of weekdays number (from [0, 6]) that should be interpreted as special ones, if given add column
            "special_days_in_week" with flag that shows given date is a special day
        special_days_in_month:
            list of days number (from [1, 31]) that should be interpreted as special ones, if given add column
            "special_days_in_month" with flag that shows given date is a special day

        Notes
        -----
        Small example of week_number_in_month and week_number_in_year features

        =============  ======================  ========================  ========================
          timestamp      day_number_in_week      week_number_in_month      week_number_in_year
        =============  ======================  ========================  ========================
        2020-01-01     4                       1                         53
        2020-01-02     5                       1                         53
        2020-01-03     6                       1                         53
        2020-01-04     0                       2                         1
        ...
        2020-01-10     6                       2                         1
        2020-01-11     0                       3                         2
        =============  ======================  ========================  ========================
        """
        if not any(
            [
                day_number_in_week,
                day_number_in_month,
                week_number_in_month,
                week_number_in_year,
                month_number_in_year,
                year_number,
                special_days_in_week,
                special_days_in_month,
            ]
        ):
            raise ValueError(
                f"{type(self).__name__} feature does nothing with given init args configuration, "
                f"at least one of day_number_in_week, day_number_in_month, week_number_in_month, "
                f"week_number_in_year, month_number_in_year, year_number should be True or any of "
                f"specyal_days_in_week, special_days_in_month should be not empty."
            )

        self.day_number_in_week = day_number_in_week
        self.day_number_in_month = day_number_in_month
        self.week_number_in_month = week_number_in_month
        self.week_number_in_year = week_number_in_year
        self.month_number_in_year = month_number_in_year
        self.year_number = year_number

        self.special_days_in_week = special_days_in_week
        self.special_days_in_month = special_days_in_month

    def fit(self, *args) -> "DateFlagsTransform":
        """Fit model. In this case of DateFlags does nothing."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get required features from df.

        Parameters
        ----------
        df:
            dataframe for feature extraction, should contain 'timestamp' column

        Returns
        -------
        dataframe with extracted features
        """
        features = pd.DataFrame(index=df.index)
        timestamp_series = pd.Series(df.index)

        if self.day_number_in_week:
            features["day_number_in_week"] = self._get_day_number_in_week(timestamp_series=timestamp_series)

        if self.day_number_in_month:
            features["day_number_in_month"] = self._get_day_number_in_month(timestamp_series=timestamp_series)

        if self.week_number_in_month:
            features["week_number_in_month"] = self._get_week_number_in_month(timestamp_series=timestamp_series)

        if self.week_number_in_year:
            features["week_number_in_year"] = self._get_week_number_in_year(timestamp_series=timestamp_series)

        if self.month_number_in_year:
            features["month_number_in_year"] = self._get_month_number_in_year(timestamp_series=timestamp_series)

        if self.year_number:
            features["year_number"] = self._get_year(timestamp_series=timestamp_series)

        if self.special_days_in_week:
            features["special_days_in_week"] = self._get_special_day_in_week(
                special_days=self.special_days_in_week, timestamp_series=timestamp_series
            )

        if self.special_days_in_month:
            features["special_days_in_month"] = self._get_special_day_in_month(
                special_days=self.special_days_in_month, timestamp_series=timestamp_series
            )

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
    def _get_special_day_in_week(special_days: Sequence[int], timestamp_series: pd.Series) -> np.array:
        """Return array with special days marked 1.

        Accepts a list of special days IN WEEK as input and returns array where these days are marked with 1
        """
        return timestamp_series.apply(lambda x: x.weekday() in special_days).values

    @staticmethod
    def _get_special_day_in_month(special_days: Sequence[int], timestamp_series: pd.Series) -> np.array:
        """Return array with special days marked 1.

        Accepts a list of special days IN MONTH as input and returns array where these days are marked with 1
        """
        return timestamp_series.apply(lambda x: x.day in special_days).values

    @staticmethod
    def _get_day_number_in_week(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the number of the day in the week."""
        return timestamp_series.apply(lambda x: x.weekday()).values

    @staticmethod
    def _get_day_number_in_month(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the number of the day in the month."""
        return timestamp_series.apply(lambda x: x.day).values

    @staticmethod
    def _get_week_number_in_month(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the week number in the month."""

        def week_of_month(dt: pd.Timestamp) -> float:
            """Return week of month number.

            How it works:
            Each month starts with the week number 1, no matter which weekday the 1st day is, for example

            * 2021-01-01 is a Friday, we mark it as 1st week
            * 2021-01-02 is a Saturday, 1st week
            * 2021-01-03 is a Sunday, 1st week
            * 2021-01-04 is a Monday, 2nd week
            * ...
            * 2021-01-10 is a Sunday, 2nd week
            * 2021-01-11 is a Monday, 3rd week
            * ...

            """
            first_day = dt.replace(day=1)

            dom = dt.day
            adjusted_dom = dom + first_day.weekday()

            return int(ceil(adjusted_dom / 7.0))

        return timestamp_series.apply(week_of_month).values

    @staticmethod
    def _get_week_number_in_year(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the week number in the year."""
        return timestamp_series.apply(lambda x: x.weekofyear).values

    @staticmethod
    def _get_month_number_in_year(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the week number in the year."""
        return timestamp_series.apply(lambda x: x.month).values

    @staticmethod
    def _get_year(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the week number in the year."""
        return timestamp_series.apply(lambda x: x.year).values


class TimeFlagsTransform(Transform):
    """Class for holding time transform."""

    is_categorical = True

    def __init__(
        self,
        minute_in_hour_number: bool = True,
        fifteen_minutes_in_hour_number: bool = False,
        hour_number: bool = True,
        half_hour_number: bool = False,
        half_day_number: bool = False,
        one_third_day_number: bool = False,
    ):
        """Initialise class attributes.

        Parameters
        ----------
        minute_in_hour_number : bool
            True if need to compute minute_in_hour_number feature
        fifteen_minutes_in_hour_number : bool
            True if need to compute fifteen_minutes_in_hour_number feature
        hour_number : bool
            True if need to compute hour_number feature
        half_hour_number : bool
            True if need to compute half_hour_number feature
        half_day_number:  bool
            True if need to compute half_day_number feature
        one_third_day_number : bool
            True if need to compute one_third_day_number feature

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

    def fit(self, *args, **kwargs) -> "_OneModelTimeFlagsFeatures":
        """Fit datetime model."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method for features based on time.

        Parameters
        ----------
        df : pd.DataFrame
            Features dataframe with time

        Returns
        -------
        result : pd.DataFrame
            Dataframe with extracted features
        """
        features = pd.DataFrame(index=df.index)
        timestamp_series = pd.Series(df.index)

        if self.minute_in_hour_number:
            minute_in_hour_number = self._get_minute_number(timestamp_series=timestamp_series)
            features["minute_in_hour_number"] = minute_in_hour_number

        if self.fifteen_minutes_in_hour_number:
            fifteen_minutes_in_hour_number = self._get_period_in_hour(
                timestamp_series=timestamp_series, period_in_minutes=15
            )
            features["fifteen_minutes_in_hour_number"] = fifteen_minutes_in_hour_number

        if self.hour_number:
            hour_number = self._get_hour_number(timestamp_series=timestamp_series)
            features["hour_number"] = hour_number

        if self.half_hour_number:
            half_hour_number = self._get_period_in_hour(timestamp_series=timestamp_series, period_in_minutes=30)
            features["half_hour_number"] = half_hour_number

        if self.half_day_number:
            half_day_number = self._get_period_in_day(timestamp_series=timestamp_series, period_in_hours=12)
            features["half_day_number"] = half_day_number

        if self.one_third_day_number:
            one_third_day_number = self._get_period_in_day(timestamp_series=timestamp_series, period_in_hours=8)
            features["one_third_day_number"] = one_third_day_number

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
    def _get_minute_number(timestamp_series: pd.Series) -> np.array:
        """Generate array with the minute number in the hour."""
        return timestamp_series.apply(lambda x: x.minute).values

    @staticmethod
    def _get_period_in_hour(timestamp_series: pd.Series, period_in_minutes: int = 15) -> np.array:
        """Generate an array with the period number in the hour.

        Accepts a period lenght in mitunes as input and returns array where timestamps marked by period number.
        """
        return timestamp_series.apply(lambda x: x.minute // period_in_minutes).values

    @staticmethod
    def _get_hour_number(timestamp_series: pd.Series) -> np.array:
        """Generate an array with the hour number in the day."""
        return timestamp_series.apply(lambda x: x.hour).values

    @staticmethod
    def _get_period_in_day(timestamp_series: pd.DataFrame, period_in_hours: int = 12) -> np.array:
        """Generate an array with the period number in the day.

        Accepts a period lenght in hours as input and returns array where timestamps marked by period number.
        """
        return timestamp_series.apply(lambda x: x.hour // period_in_hours).values


__all__ = ["TimeFlagsTransform", "TimeFlagsTransform", "DateFlagsTransform"]
