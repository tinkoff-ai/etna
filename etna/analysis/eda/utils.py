from functools import singledispatch
from typing import TYPE_CHECKING
from typing import List
from typing import Optional

import holidays as holidays_lib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_correlation_matrix(
    ts: "TSDataset",
    columns: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    method: str = "pearson",
) -> np.ndarray:
    """Compute pairwise correlation of timeseries for selected segments.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    columns:
        Columns to use, if None use all columns
    segments:
        Segments to use
    method:
        Method of correlation:

        * pearson: standard correlation coefficient

        * kendall: Kendall Tau correlation coefficient

        * spearman: Spearman rank correlation

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if method not in ["pearson", "kendall", "spearman"]:
        raise ValueError(f"'{method}' is not a valid method of correlation.")

    if segments is None:
        segments = sorted(ts.segments)
    if columns is None:
        columns = list(set(ts.df.columns.get_level_values("feature")))

    correlation_matrix = ts[:, segments, columns].corr(method=method).values
    return correlation_matrix


@singledispatch
def _create_holidays_df(holidays, index: pd.core.indexes.datetimes.DatetimeIndex, as_is: bool) -> pd.DataFrame:
    raise ValueError("Parameter holidays is expected as str or pd.DataFrame")


@_create_holidays_df.register
def _create_holidays_df_str(holidays: str, index, as_is):
    if as_is:
        raise ValueError("Parameter `as_is` should be used with `holiday`: pd.DataFrame, not string.")
    timestamp = index.tolist()
    country_holidays = holidays_lib.country_holidays(country=holidays)
    holiday_names = {country_holidays.get(timestamp_value) for timestamp_value in timestamp}
    holiday_names = holiday_names.difference({None})

    holidays_dict = {}
    for holiday_name in holiday_names:
        cur_holiday_index = pd.Series(timestamp).apply(
            lambda x: country_holidays.get(x, "") == holiday_name  # noqa: B023
        )
        holidays_dict[holiday_name] = cur_holiday_index

    holidays_df = pd.DataFrame(holidays_dict)
    holidays_df.index = timestamp
    return holidays_df


@_create_holidays_df.register
def _create_holidays_df_dataframe(holidays: pd.DataFrame, index, as_is):
    if holidays.empty:
        raise ValueError("Got empty `holiday` pd.DataFrame.")

    if as_is:
        holidays_df = pd.DataFrame(index=index, columns=holidays.columns, data=False)
        dt = holidays_df.index.intersection(holidays.index)
        holidays_df.loc[dt, :] = holidays.loc[dt, :]
        return holidays_df

    holidays_df = pd.DataFrame(index=index, columns=holidays["holiday"].unique(), data=False)
    for name in holidays["holiday"].unique():
        freq = pd.infer_freq(index)
        ds = holidays[holidays["holiday"] == name]["ds"]
        dt = [ds]
        if "upper_window" in holidays.columns:
            periods = holidays[holidays["holiday"] == name]["upper_window"].fillna(0).tolist()[0]
            if periods < 0:
                raise ValueError("Upper windows should be non-negative.")
            ds_upper_bound = pd.timedelta_range(start=0, periods=periods + 1, freq=freq)
            for bound in ds_upper_bound:
                ds_add = ds + bound
                dt.append(ds_add)
        if "lower_window" in holidays.columns:
            periods = holidays[holidays["holiday"] == name]["lower_window"].fillna(0).tolist()[0]
            if periods > 0:
                raise ValueError("Lower windows should be non-positive.")
            ds_lower_bound = pd.timedelta_range(start=0, periods=abs(periods) + 1, freq=freq)
            for bound in ds_lower_bound:
                ds_add = ds - bound
                dt.append(ds_add)
        dt = pd.concat(dt)
        dt = holidays_df.index.intersection(dt)
        holidays_df.loc[dt, name] = True
    return holidays_df
