from copy import deepcopy
from math import ceil
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import Transform


class DateFlagsTransform(Transform, FutureMixin):
    """DateFlagsTransform is a class that implements extraction of the main date-based features from datetime column.

    Notes
    -----
    Small example of ``week_number_in_month`` and ``week_number_in_year`` features

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

    def __init__(
        self,
        day_number_in_week: Optional[bool] = True,
        day_number_in_month: Optional[bool] = True,
        day_number_in_year: Optional[bool] = False,
        week_number_in_month: Optional[bool] = False,
        week_number_in_year: Optional[bool] = False,
        month_number_in_year: Optional[bool] = False,
        season_number: Optional[bool] = False,
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
        day_number_in_year:
            if True, add column with number of day in a year with leap year numeration (values from 1 to 366)
        week_number_in_month:
            if True, add column with week number (in month context) to feature dataframe in transform
        week_number_in_year:
            if True, add column with week number (in year context) to feature dataframe in transform
        month_number_in_year:
            if True, add column with month info to feature dataframe in transform
        season_number:
            if True, add column with season info to feature dataframe in transform
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
            base for the name of created columns;

            * if set the final name is '{out_column}_{feature_name}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        """
        if not any(
            [
                day_number_in_week,
                day_number_in_month,
                day_number_in_year,
                week_number_in_month,
                week_number_in_year,
                month_number_in_year,
                season_number,
                year_number,
                is_weekend,
                special_days_in_week,
                special_days_in_month,
            ]
        ):
            raise ValueError(
                f"{type(self).__name__} feature does nothing with given init args configuration, "
                f"at least one of day_number_in_week, day_number_in_month, day_number_in_year, week_number_in_month, "
                f"week_number_in_year, month_number_in_year, season_number, year_number, is_weekend should be True or any of "
                f"special_days_in_week, special_days_in_month should be not empty."
            )

        self.day_number_in_week = day_number_in_week
        self.day_number_in_month = day_number_in_month
        self.day_number_in_year = day_number_in_year
        self.week_number_in_month = week_number_in_month
        self.week_number_in_year = week_number_in_year
        self.month_number_in_year = month_number_in_year
        self.season_number = season_number
        self.year_number = year_number
        self.is_weekend = is_weekend

        self.special_days_in_week = special_days_in_week
        self.special_days_in_month = special_days_in_month

        self.out_column = out_column

        # create empty init parameters
        self._empty_parameters = dict(
            day_number_in_week=False,
            day_number_in_month=False,
            day_number_in_year=False,
            week_number_in_month=False,
            week_number_in_year=False,
            month_number_in_year=False,
            season_number=False,
            year_number=False,
            is_weekend=False,
            special_days_in_week=(),
            special_days_in_month=(),
        )

    def _get_column_name(self, feature_name: str) -> str:
        if self.out_column is None:
            init_parameters = deepcopy(self._empty_parameters)
            init_parameters[feature_name] = self.__dict__[feature_name]
            temp_transform = DateFlagsTransform(**init_parameters, out_column=self.out_column)  # type: ignore
            return temp_transform.__repr__()
        else:
            return f"{self.out_column}_{feature_name}"

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
        :
            dataframe with extracted features
        """
        features = pd.DataFrame(index=df.index)
        timestamp_series = pd.Series(df.index)

        if self.day_number_in_week:
            features[self._get_column_name("day_number_in_week")] = self._get_day_number_in_week(
                timestamp_series=timestamp_series
            )

        if self.day_number_in_month:
            features[self._get_column_name("day_number_in_month")] = self._get_day_number_in_month(
                timestamp_series=timestamp_series
            )

        if self.day_number_in_year:
            features[self._get_column_name("day_number_in_year")] = self._get_day_number_in_year(
                timestamp_series=timestamp_series
            )

        if self.week_number_in_month:
            features[self._get_column_name("week_number_in_month")] = self._get_week_number_in_month(
                timestamp_series=timestamp_series
            )

        if self.week_number_in_year:
            features[self._get_column_name("week_number_in_year")] = self._get_week_number_in_year(
                timestamp_series=timestamp_series
            )

        if self.month_number_in_year:
            features[self._get_column_name("month_number_in_year")] = self._get_month_number_in_year(
                timestamp_series=timestamp_series
            )

        if self.season_number:
            features[self._get_column_name("season_number")] = self._get_season_number(
                timestamp_series=timestamp_series
            )

        if self.year_number:
            features[self._get_column_name("year_number")] = self._get_year(timestamp_series=timestamp_series)

        if self.is_weekend:
            features[self._get_column_name("is_weekend")] = self._get_weekends(timestamp_series=timestamp_series)

        if self.special_days_in_week:
            features[self._get_column_name("special_days_in_week")] = self._get_special_day_in_week(
                special_days=self.special_days_in_week, timestamp_series=timestamp_series
            )

        if self.special_days_in_month:
            features[self._get_column_name("special_days_in_month")] = self._get_special_day_in_month(
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
    def _get_season_number(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the season number."""
        return timestamp_series.apply(lambda x: x.month % 12 // 3 + 1).values

    @staticmethod
    def _get_day_number_in_year(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with number of day in a year with leap year numeration (values from 1 to 366)."""

        def leap_year_number(dt: pd.Timestamp) -> int:
            """Return day number with leap year numeration."""
            day_of_year = dt.dayofyear
            if not dt.is_leap_year and dt.month >= 3:
                return day_of_year + 1
            else:
                return day_of_year

        return timestamp_series.apply(leap_year_number).values

    @staticmethod
    def _get_week_number_in_month(timestamp_series: pd.Series) -> np.ndarray:
        """Generate an array with the week number in the month."""

        def week_of_month(dt: pd.Timestamp) -> int:
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


__all__ = ["DateFlagsTransform"]
