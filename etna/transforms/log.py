import warnings
from math import log
from math import pow
from typing import Optional

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentLogTransform(Transform):
    """Instance of this class applies logarithmic transformation to one segment data."""

    def __init__(self, in_column: str, base: int = 10, inplace: bool = True, out_column: Optional[str] = None):
        """
        Init OneSegmentLogTransform.

        Parameters
        ----------
        in_column:
            column to apply transform.
        base:
            base of logarithm to apply to series.
        inplace:
            if True, apply logarithm transformation inplace to in_column, if False, add transformed column to dataset.
        out_column:
            name of added column. If not given, use self.__repr__()
        """
        self.base = base
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column

    def fit(self, df: pd.Series) -> "_OneSegmentLogTransform":
        """Fit preprocess method, does nothing in OneSegmentLogTransform case.

        Returns
        -------
        self
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to series from df.

        Parameters
        ----------
        df:
            series to transform

        Returns
        -------
        transformed series

        Raises
        ------
        ValueError:
            if given series contains negative samples
        """
        if (df[self.in_column] < 0).any():
            raise ValueError("LogPreprocess can be applied only to non-negative series")
        result_df = df.copy()
        result_df[self.out_column] = result_df[self.in_column].apply(lambda x: log(x + 1, self.base))
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to the series from df.

        Parameters
        ----------
        df:
            series to transform

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

    def __init__(self, in_column: str, base: int = 10, inplace: bool = True, out_column: Optional[str] = None):
        """Init LogTransform.

        Parameters
        ----------
        in_column:
            column to apply transform
        base:
            base of logarithm to apply to series
        inplace:
            if True, apply logarithm transformation inplace to in_column,
            if False, add column add transformed column to dataset
        out_column:
            name of added column. If not given, use self.__repr__()
        """
        self.in_column = in_column
        self.base = base
        self.inplace = inplace
        self.out_column = out_column
        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")

        if self.inplace:
            result_out_column = self.in_column
        elif out_column:
            result_out_column = out_column
        else:
            result_out_column = self.__repr__()

        super().__init__(
            transform=_OneSegmentLogTransform(
                in_column=in_column, base=base, inplace=inplace, out_column=result_out_column
            )
        )


__all__ = ["LogTransform"]
