import datetime

import holidays
import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentHolidayTransform(Transform):
    """Mark holidays as 1 and usual days as 0."""

    def __init__(self, iso_code: str = "RUS"):
        """
        Create instance of _OneSegmentHolidayTransform.
        Parameters
        ----------
        iso_code:
            internationally recognised codes, designated to country for which we want to find the holidays
        """
        self.holidays = holidays.CountryHoliday(iso_code)
        self.out_prefix = "regressor_"

    def fit(self, df: pd.DataFrame) -> "_OneSegmentHolidayTransform":
        """
        Fit _OneSegmentHolidayTransform with data from df. Does nothing in this case.
        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data from df with _OneSegmentHolidayTransform and generate a column of holidays flags.
        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format
        Returns
        -------
            pd.DataFrame with 'holidays' column
        """
        if (df.index[1] - df.index[0]) > datetime.timedelta(days=1):
            raise ValueError("Frequency of data should be no more than daily.")

        timestamp_df = df.reset_index()["timestamp"]

        to_add = pd.DataFrame()
        to_add["holidays"] = timestamp_df.apply(lambda x: int(x in self.holidays)).astype("category")

        to_add = to_add.add_prefix(self.out_prefix)
        to_add.index = df.index
        to_return = df.copy()
        to_return = pd.concat([to_return, to_add], axis=1)
        to_return.columns.names = df.columns.names
        return to_return


class HolidayTransform(PerSegmentWrapper):
    """HolidayTransform generates series that indicates holidays in given dataframe. Creates column 'holidays'."""

    def __init__(self, iso_code: str = "RUS"):
        """
        Create instance of HolidayTransform.
        Parameters
        ----------
        iso_code:
            internationally recognised codes, designated to country for which we want to find the holidays
        """
        self.iso_code = iso_code
        super().__init__(transform=_OneSegmentHolidayTransform(self.iso_code))
