import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentAddConstTransform(Transform):
    def __init__(self, value: float, in_column: str, inplace: bool = True):
        """
        Init _OneSegmentAddConstTransform.

        Parameters
        ----------
        value:
            value that should be added to the series
        in_column:
            column to apply transform
        inplace:
            if True, apply add constant transformation inplace to in_column, if False, add column {in_column}_add_{value} to dataset
        """
        self.value = value
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = self.in_column if self.inplace else f"{self.in_column}_add_{self.value}"

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

    def __init__(self, value: float, in_column: str, inplace: bool = True):
        """
        Init AddConstTransform.

        Parameters
        ----------
        value:
            value that should be added to the series
        in_column:
            column to apply transform
        inplace:
            if True, apply add constant transformation inplace to in_column, if False, add column {in_column}_add_{value} to dataset
        """
        self.value = value
        self.in_column = in_column
        self.inplace = inplace
        super().__init__(
            transform=_OneSegmentAddConstTransform(value=self.value, in_column=self.in_column, inplace=self.inplace)
        )


__all__ = ["AddConstTransform"]
