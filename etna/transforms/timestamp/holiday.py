import datetime
from typing import Optional

import holidays
import numpy as np
import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import Transform


class HolidayTransform(Transform, FutureMixin):
    """HolidayTransform generates series that indicates holidays in given dataframe."""

    def __init__(self, iso_code: str = "RUS", out_column: Optional[str] = None):
        """
        Create instance of HolidayTransform.

        Parameters
        ----------
        iso_code:
            internationally recognised codes, designated to country for which we want to find the holidays
        out_column:
            name of added column. Use ``self.__repr__()`` if not given.
        """
        self.iso_code = iso_code
        self.holidays = holidays.CountryHoliday(iso_code)
        self.out_column = out_column
        self.out_column = self.out_column if self.out_column is not None else self.__repr__()

    def fit(self, df: pd.DataFrame) -> "HolidayTransform":
        """
        Fit HolidayTransform with data from df. Does nothing in this case.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data from df with HolidayTransform and generate a column of holidays flags.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format

        Returns
        -------
        :
            pd.DataFrame with added holidays
        """
        if (df.index[1] - df.index[0]) > datetime.timedelta(days=1):
            raise ValueError("Frequency of data should be no more than daily.")

        cols = df.columns.get_level_values("segment").unique()

        encoded_matrix = np.array([int(x in self.holidays) for x in df.index])
        encoded_matrix = encoded_matrix.reshape(-1, 1).repeat(len(cols), axis=1)
        encoded_df = pd.DataFrame(
            encoded_matrix,
            columns=pd.MultiIndex.from_product([cols, [self.out_column]], names=("segment", "feature")),
            index=df.index,
        )
        encoded_df = encoded_df.astype("category")

        df = df.join(encoded_df)
        df = df.sort_index(axis=1)
        return df
