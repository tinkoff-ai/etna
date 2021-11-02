import warnings
from typing import Optional

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentAddConstTransform(Transform):
    def __init__(self, in_column: str, out_column: str, value: float, inplace: bool = True):
        """
        Init _OneSegmentAddConstTransform.

        Parameters
        ----------
        in_column:
            column to apply transform
        out_column:
            name of added column
        value:
            value that should be added to the series
        inplace:
            if True, apply add constant transformation inplace to in_column, if False, add transformed column to dataset
        """
        self.in_column = in_column
        self.value = value
        self.inplace = inplace
        self.out_column = out_column

    def fit(self, df: pd.DataFrame) -> "_OneSegmentAddConstTransform":
        """
        Fit preprocess method, does nothing in _OneSegmentAddConstTransform case.

        Returns
        -------
        self
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method: add given value to all the series values.

        Parameters
        ----------
        df:
            DataFrame to transform

        Returns
        -------
        transformed series
        """
        result_df = df.copy()
        result_df[self.out_column] = result_df[self.in_column] + self.value
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation: subtract given value from all the series values.

        Parameters
        ----------
        df:
            DataFrame to apply inverse transformation

        Returns
        -------
        transformed series
        """
        result_df = df.copy()
        if self.inplace:
            result_df[self.in_column] = result_df[self.out_column] - self.value
        return result_df


class AddConstTransform(PerSegmentWrapper):
    """AddConstTransform add constant for given series."""

    def __init__(self, in_column: str, value: float, inplace: bool = True, out_column: Optional[str] = None):
        """
        Init AddConstTransform.

        Parameters
        ----------
        in_column:
            column to apply transform
        value:
            value that should be added to the series
        inplace:
            if True, apply add constant transformation inplace to in_column, if False, add transformed column to dataset
        out_column:
            name of added column.Don't forget to add regressor prefix if necessary. If not given, use self.__repr__()
        """
        self.in_column = in_column
        self.value = value
        self.inplace = inplace
        self.out_column = out_column

        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")

        if self.inplace:
            out_column_result = self.in_column
        elif self.out_column:
            out_column_result = self.out_column
        else:
            out_column_result = self.__repr__()
        super().__init__(
            transform=_OneSegmentAddConstTransform(
                in_column=in_column, value=value, inplace=inplace, out_column=out_column_result
            )
        )


__all__ = ["AddConstTransform"]
