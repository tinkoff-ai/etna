import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import FutureMixin
from etna.transforms.base import IrreversiblePerSegmentWrapper
from etna.transforms.base import OneSegmentTransform


def calc_day_number_in_week(datetime_day: datetime.datetime) -> int:
    return datetime_day.weekday()


def calc_day_number_in_month(datetime_day: datetime.datetime) -> int:
    return datetime_day.day


class _OneSegmentSpecialDaysTransform(OneSegmentTransform):
    """
    Search for anomalies in values, marked this days as 1 (and return new column with 1 in corresponding places).

    Notes
    -----
    You can read more about other anomalies detection methods in:
    `Time Series of Price Anomaly Detection <https://towardsdatascience.com/time-series-of-price-anomaly-detection-13586cd5ff46>`_
    """

    def __init__(self, find_special_weekday: bool = True, find_special_month_day: bool = True):
        """
        Create instance of _OneSegmentSpecialDaysTransform.

        Parameters
        ----------
        find_special_weekday:
            flag, if True, find special weekdays in transform
        find_special_month_day:
            flag, if True, find special monthdays in transform

        Raises
        ------
        ValueError:
            if all the modes are False
        """
        if not any([find_special_weekday, find_special_month_day]):
            raise ValueError(
                f"{type(self).__name__} feature does nothing with given init args configuration, "
                f"at least one of find_special_weekday, find_special_month_day should be True."
            )

        self.find_special_weekday = find_special_weekday
        self.find_special_month_day = find_special_month_day

        self.anomaly_week_days: Optional[Tuple[int]] = None
        self.anomaly_month_days: Optional[Tuple[int]] = None

        self.res_type: Dict[str, Any]
        if self.find_special_weekday and find_special_month_day:
            self.res_type = {"df_sample": (0, 0), "columns": ["anomaly_weekdays", "anomaly_monthdays"]}
        elif self.find_special_weekday:
            self.res_type = {"df_sample": 0, "columns": ["anomaly_weekdays"]}
        elif self.find_special_month_day:
            self.res_type = {"df_sample": 0, "columns": ["anomaly_monthdays"]}
        else:
            raise ValueError("nothing to do")

    def fit(self, df: pd.DataFrame) -> "_OneSegmentSpecialDaysTransform":
        """
        Fit _OneSegmentSpecialDaysTransform with data from df.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format
        """
        common_df = df[["target"]].reset_index()
        common_df.columns = ["datetime", "value"]

        if self.find_special_weekday:
            self.anomaly_week_days = self._find_anomaly_day_in_week(common_df)

        if self.find_special_month_day:
            self.anomaly_month_days = self._find_anomaly_day_in_month(common_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data from df with _OneSegmentSpecialDaysTransform and generate a column of special day flags.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format

        Returns
        -------
        :
            pd.DataFrame with 'anomaly_weekday', 'anomaly_monthday' or both of them columns no-timestamp indexed that
            contains 1 at i-th position if i-th day is a special day
        """
        common_df = df[["target"]].reset_index()
        common_df.columns = ["datetime", "value"]

        to_add = pd.DataFrame([self.res_type["df_sample"]] * len(df), columns=self.res_type["columns"])

        if self.find_special_weekday:
            if self.anomaly_week_days is None:
                raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
            to_add["anomaly_weekdays"] += self._marked_special_week_day(common_df, self.anomaly_week_days)
            to_add["anomaly_weekdays"] = to_add["anomaly_weekdays"].astype("category")

        if self.find_special_month_day:
            if self.anomaly_month_days is None:
                raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
            to_add["anomaly_monthdays"] += self._marked_special_month_day(common_df, self.anomaly_month_days)
            to_add["anomaly_monthdays"] = to_add["anomaly_monthdays"].astype("category")

        to_add.index = df.index
        to_return = pd.concat([df, to_add], axis=1)
        to_return.columns.names = df.columns.names
        return to_return

    @staticmethod
    def _find_anomaly_day_in_week(df: pd.DataFrame, agg_func=pd.core.groupby.SeriesGroupBy.mean) -> Tuple[int]:
        cp_df = df.copy()

        cp_df = pd.concat(
            [cp_df, cp_df["datetime"].apply(calc_day_number_in_week).rename("weekday").astype(int)], axis=1
        )
        cp_df = cp_df.groupby(["weekday"])

        t = agg_func((cp_df[["value"]])).quantile(q=0.95).tolist()[0]

        return cp_df.filter(lambda x: x["value"].mean() > t).loc[:, "weekday"].tolist()

    @staticmethod
    def _find_anomaly_day_in_month(df: pd.DataFrame, agg_func=pd.core.groupby.SeriesGroupBy.mean) -> Tuple[int]:
        cp_df = df.copy()

        cp_df = pd.concat(
            [cp_df, cp_df["datetime"].apply(calc_day_number_in_month).rename("monthday").astype(int)], axis=1
        )
        cp_df = cp_df.groupby(["monthday"])

        t = agg_func(cp_df[["value"]]).quantile(q=0.95).tolist()[0]

        return cp_df.filter(lambda x: x["value"].mean() > t).loc[:, "monthday"].tolist()

    @staticmethod
    def _marked_special_week_day(df: pd.DataFrame, week_days: Tuple[int]) -> pd.Series:
        """Mark desired week day in dataframe, return column with original length."""

        def check(x):
            return calc_day_number_in_week(x["datetime"]) in week_days

        return df.loc[:, ["datetime"]].apply(check, axis=1).rename("anomaly_weekdays")

    @staticmethod
    def _marked_special_month_day(df: pd.DataFrame, month_days: Tuple[int]) -> pd.Series:
        """Mark desired month day in dataframe, return column with original length."""

        def check(x):
            return calc_day_number_in_month(x["datetime"]) in month_days

        return df.loc[:, ["datetime"]].apply(check, axis=1).rename("anomaly_monthdays")

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform Dataframe."""
        return df


class SpecialDaysTransform(IrreversiblePerSegmentWrapper, FutureMixin):
    """SpecialDaysTransform generates series that indicates is weekday/monthday is special in given dataframe.

    Creates columns 'anomaly_weekdays' and 'anomaly_monthdays'.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(self, find_special_weekday: bool = True, find_special_month_day: bool = True):
        """
        Create instance of SpecialDaysTransform.

        Parameters
        ----------
        find_special_weekday:
            flag, if True, find special weekdays in transform
        find_special_month_day:
            flag, if True, find special monthdays in transform

        Raises
        ------
        ValueError:
            if all the modes are False
        """
        self.find_special_weekday = find_special_weekday
        self.find_special_month_day = find_special_month_day
        super().__init__(
            transform=_OneSegmentSpecialDaysTransform(self.find_special_weekday, self.find_special_month_day),
            required_features=["target"],
        )

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        output_columns = []
        if self.find_special_weekday:
            output_columns.append("anomaly_weekdays")
        if self.find_special_month_day:
            output_columns.append("anomaly_monthdays")
        return output_columns

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``find_special_weekday``, ``find_special_month_day``.
        Other parameters are expected to be set by the user.

        There are no restrictions on all ``False`` values for the flags.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "find_special_weekday": CategoricalDistribution([False, True]),
            "find_special_month_day": CategoricalDistribution([False, True]),
        }


__all__ = ["SpecialDaysTransform"]
