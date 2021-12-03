from math import ceil
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd

from etna.transforms.base import Transform


class DateFlagsTransform(Transform):
    """DateFlagsTransform is a class that implements extraction of the main date-based features from datetime column.
    Creates columns named '{out_column}_{feature_name}'(don't forget to add regressor prefix if necessary)
    or 'regressor_{__repr__()}_{feature_name}' if not given.

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
        is_weekend: Optional[bool] = True,
        special_days_in_week: Sequence[int] = (),
        special_days_in_month: Sequence[int] = (),
        out_column: Optional[str] = None,
    ):
        """Create instance of DateFlags.

        Parameters
        ----------
        day_number_in_week:
            if True, add column with weekday info to feature dataframe in transform
        day_number_in_month:
            if True, add column with day info to feature dataframe in transform
        week_number_in_month:
            if True, add column with week number (in month context) to feature dataframe in transform
        week_number_in_year:
            if True, add column with week number (in year context) to feature dataframe in transform
        month_number_in_year:
            if True, add column with month info to feature dataframe in transform
        year_number:
            if True, add column with year info to feature dataframe in transform
        is_weekend:
            if True: add column with weekends flags to feature dataframe in transform
        special_days_in_week:
            list of weekdays number (from [0, 6]) that should be interpreted as special ones, if given add column
            with flag that shows given date is a special day
        special_days_in_month:
            list of days number (from [1, 31]) that should be interpreted as special ones, if given add column
            with flag that shows given date is a special day
        out_column:
            name of added column. We get '{out_column}_{feature_name}'.
            If not given, use 'regressor_{self.__repr__()}_{feature_name}'

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
                is_weekend,
                special_days_in_week,
                special_days_in_month,
            ]
        ):
            raise ValueError(
                f"{type(self).__name__} feature does nothing with given init args configuration, "
                f"at least one of day_number_in_week, day_number_in_month, week_number_in_month, "
                f"week_number_in_year, month_number_in_year, year_number, is_weekend should be True or any of "
                f"special_days_in_week, special_days_in_month should be not empty."
            )

        self.day_number_in_week = day_number_in_week
        self.day_number_in_month = day_number_in_month
        self.week_number_in_month = week_number_in_month
        self.week_number_in_year = week_number_in_year
        self.month_number_in_year = month_number_in_year
        self.year_number = year_number
        self.is_weekend = is_weekend

        self.special_days_in_week = special_days_in_week
        self.special_days_in_month = special_days_in_month

        self.out_column = out_column
        self.out_column_prefix = out_column if out_column is not None else f"regressor_{self.__repr__()}"

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
            features[f"{self.out_column_prefix}_day_number_in_week"] = self._get_day_number_in_week(
                timestamp_series=timestamp_series
            )

        if self.day_number_in_month:
            features[f"{self.out_column_prefix}_day_number_in_month"] = self._get_day_number_in_month(
                timestamp_series=timestamp_series
            )

        if self.week_number_in_month:
            features[f"{self.out_column_prefix}_week_number_in_month"] = self._get_week_number_in_month(
                timestamp_series=timestamp_series
            )

        if self.week_number_in_year:
            features[f"{self.out_column_prefix}_week_number_in_year"] = self._get_week_number_in_year(
                timestamp_series=timestamp_series
            )

        if self.month_number_in_year:
            features[f"{self.out_column_prefix}_month_number_in_year"] = self._get_month_number_in_year(
                timestamp_series=timestamp_series
            )

        if self.year_number:
            features[f"{self.out_column_prefix}_year_number"] = self._get_year(timestamp_series=timestamp_series)

        if self.is_weekend:
            features[f"{self.out_column_prefix}_is_weekend"] = self._get_weekends(timestamp_series=timestamp_series)

        if self.special_days_in_week:
            features[f"{self.out_column_prefix}_special_days_in_week"] = self._get_special_day_in_week(
                special_days=self.special_days_in_week, timestamp_series=timestamp_series
            )

        if self.special_days_in_month:
            features[f"{self.out_column_prefix}_special_days_in_month"] = self._get_special_day_in_month(
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
    def _get_special_day_in_week(special_days: Sequence[int], timestamp_series: pd.Series) -> np.ndarray:
        """Return array with special days marked 1.

        Accepts a list of special days IN WEEK as input and returns array where these days are marked with 1
        """
        return timestamp_series.apply(lambda x: x.weekday() in special_days).values

    @staticmethod
    def _get_special_day_in_month(special_days: Sequence[int], timestamp_series: pd.Series) -> np.ndarray:
        """Return array with special days marked 1.

        Accepts a list of special days IN MONTH as input and returns array where these days are marked with 1
        """
        return timestamp_series.apply(lambda x: x.day in special_days).values

    @staticmethod
    def _get_day_number_in_week(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the number of the day in the week."""
        return timestamp_series.apply(lambda x: x.weekday()).values

    @staticmethod
    def _get_day_number_in_month(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the number of the day in the month."""
        return timestamp_series.apply(lambda x: x.day).values

    @staticmethod
    def _get_week_number_in_month(timestamp_series: pd.Series) -> np.ndarray:
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
    def _get_week_number_in_year(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the week number in the year."""
        return timestamp_series.apply(lambda x: x.weekofyear).values

    @staticmethod
    def _get_month_number_in_year(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the week number in the year."""
        return timestamp_series.apply(lambda x: x.month).values

    @staticmethod
    def _get_year(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the week number in the year."""
        return timestamp_series.apply(lambda x: x.year).values

    @staticmethod
    def _get_weekends(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the weekends flags."""
        weekend_days = (5, 6)
        return timestamp_series.apply(lambda x: x.weekday() in weekend_days).values


class TimeFlagsTransform(Transform):
    """Class for holding time transform.
    Creates columns named '{out_column}_{feature_name}'(don't forget to add regressor prefix if necessary)
    or 'regressor_{__repr__()}_{feature_name}' if not given.
    """

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
            if True: add column with weekends flags to feature dataframe in transform
        fifteen_minutes_in_hour_number:
            if True: add column with weekends flags
            to feature dataframe in transform
        hour_number:
            if True: add column with weekends flags to feature dataframe in transform
        half_hour_number:
            if True: add column with weekends flags to feature dataframe in transform
        half_day_number:
            if True: add column with weekends flags to feature dataframe in transform
        one_third_day_number:
            if True: add column with weekends flags to feature dataframe in transform
        out_column:
            name of added column. We get '{out_column}_{feature_name}'.
            If not given, use 'regressor_{self.__repr__()}_{feature_name}'

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
        self.out_column_prefix = out_column if out_column is not None else f"regressor_{self.__repr__()}"

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
            features[f"{self.out_column_prefix}_minute_in_hour_number"] = minute_in_hour_number

        if self.fifteen_minutes_in_hour_number:
            fifteen_minutes_in_hour_number = self._get_period_in_hour(
                timestamp_series=timestamp_series, period_in_minutes=15
            )
            features[f"{self.out_column_prefix}_fifteen_minutes_in_hour_number"] = fifteen_minutes_in_hour_number

        if self.hour_number:
            hour_number = self._get_hour_number(timestamp_series=timestamp_series)
            features[f"{self.out_column_prefix}_hour_number"] = hour_number

        if self.half_hour_number:
            half_hour_number = self._get_period_in_hour(timestamp_series=timestamp_series, period_in_minutes=30)
            features[f"{self.out_column_prefix}_half_hour_number"] = half_hour_number

        if self.half_day_number:
            half_day_number = self._get_period_in_day(timestamp_series=timestamp_series, period_in_hours=12)
            features[f"{self.out_column_prefix}_half_day_number"] = half_day_number

        if self.one_third_day_number:
            one_third_day_number = self._get_period_in_day(timestamp_series=timestamp_series, period_in_hours=8)
            features[f"{self.out_column_prefix}_one_third_day_number"] = one_third_day_number

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


__all__ = ["TimeFlagsTransform", "DateFlagsTransform"]
