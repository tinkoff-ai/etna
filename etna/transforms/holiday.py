import holidays
import numpy as np
import pandas as pd

from etna.transforms.base import Transform


class HolidayTransform(Transform):
    """HolidayTransform generates series that indicates holidays in given dataframe. Creates column 'regressor_holidays'."""

    def __init__(self, iso_code: str = "RUS"):
        """
        Create instance of HolidayTransform.
        Parameters
        ----------
        iso_code:
            internationally recognised codes, designated to country for which we want to find the holidays
        """
        self.holidays = holidays.CountryHoliday(iso_code)

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
            pd.DataFrame with 'regressor_holidays' column
        """
        timestamp_df = df.reset_index()["timestamp"]
        encoded_matrix = np.array(timestamp_df.apply(lambda x: int(x in self.holidays)).astype("category"))

        cols = df.columns.get_level_values("segment")
        encoded_matrix = encoded_matrix.reshape(-1, 1).repeat(len(cols), axis=1)
        encoded_df = pd.DataFrame(
            encoded_matrix,
            columns=pd.MultiIndex.from_product([cols, ["regressor_holidays"]], names=("segment", "feature")),
            index=df.index,
        )
        encoded_df = encoded_df.astype("category")

        df = df.join(encoded_df)
        df = df.sort_index(axis=1)
        return df
