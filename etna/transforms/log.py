from math import log
from math import pow

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentLogTransform(Transform):
    """Instance of this class applies logarithmic transformation to one segment data."""

    def __init__(self, in_column: str, base: int = 10, inplace: bool = True):
        """
        Init OneSegmentLogTransform.

        Parameters
        ----------
        in_column:
            column to apply transform.
        base:
            base of logarithm to apply to series.
        inplace:
            if True, apply logarithm transformation inplace to in_column, if False, add column {in_column}_log_{base} to dataset.
        """
        self.base = base
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = self.in_column if self.inplace else f"{self.in_column}_log_{self.base}"

    def fit(self, df: pd.Series) -> "_OneSegmentLogTransform":
        """
        Fit preprocess method, does nothing in OneSegmentLogTransform case.

        Returns
        -------
        self
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation to series from df.

        Parameters
        ----------
        df:
            series to transform.

        Returns
        -------
        transformed series

        Raises
        ------
        ValueError:
            if given series contains negative samples.
        """
        if (df[self.in_column] < 0).any():
            raise ValueError("LogPreprocess can be applied only to non-negative series")
        result_df = df.copy()
        result_df[self.out_column] = result_df[self.in_column].apply(lambda x: log(x + 1, self.base))
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation to the series from df.

        Parameters
        ----------
        df:
            series to transform.

        Returns
        -------
        transformed series
        """
        result_df = df.copy()
        if self.inplace:
            result_df[self.in_column] = result_df[self.out_column].apply(lambda x: pow(self.base, x) - 1)
        return result_df


class LogTransform(PerSegmentWrapper):
    """LogTransform applies logarithm transformation for given series."""

    def __init__(self, in_column: str, base: int = 10, inplace: bool = True):
        """
        Init LogTransform.

        Parameters
        ----------
        in_column:
            column to apply transform.
        base:
            base of logarithm to apply to series.
        inplace:
            if True, apply logarithm transformation inplace to in_column,
            if False, add column {in_column}_log_{base} to dataset.
        """
        self.in_column = in_column
        self.base = base
        self.inplace = inplace
        super().__init__(
            transform=_OneSegmentLogTransform(in_column=self.in_column, base=self.base, inplace=self.inplace)
        )


__all__ = ["LogTransform"]
