import datetime
from typing import List
from typing import Optional

import holidays
import numpy as np
import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import IrreversibleTransform


class HolidayTransform(IrreversibleTransform, FutureMixin):
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
        super().__init__(required_features=["target"])
        self.iso_code = iso_code
        self.holidays = holidays.CountryHoliday(iso_code)
        self.out_column = out_column

    def _get_column_name(self) -> str:
        if self.out_column:
            return self.out_column
        else:
            return self.__repr__()

    def _fit(self, df: pd.DataFrame) -> "HolidayTransform":
        """
        Fit HolidayTransform with data from df. Does nothing in this case.

        Parameters
        ----------
        df: pd.DataFrame
            value series with index column in timestamp format
        """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

        out_column = self._get_column_name()
        encoded_matrix = np.array([int(x in self.holidays) for x in df.index])
        encoded_matrix = encoded_matrix.reshape(-1, 1).repeat(len(cols), axis=1)
        encoded_df = pd.DataFrame(
            encoded_matrix,
            columns=pd.MultiIndex.from_product([cols, [out_column]], names=("segment", "feature")),
            index=df.index,
        )
        encoded_df = encoded_df.astype("category")

        df = df.join(encoded_df)
        df = df.sort_index(axis=1)
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.
        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return [self._get_column_name()]
