import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentAddConstTransform(Transform):
    def __init__(self, value: float, in_column: str, inplace: bool = True, out_column: str = None):
        """
        Init _OneSegmentAddConstTransform.

        Parameters
        ----------
        value:
            value that should be added to the series
        in_column:
            column to apply transform
        inplace:
            if True, apply add constant transformation inplace to in_column, if False, add transformed column to dataset
        out_column:
            name of added column
        """
        self.value = value
        self.in_column = in_column
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

    def __init__(self, value: float, in_column: str, inplace: bool = True, out_column: str = None):
        """
        Init AddConstTransform.

        Parameters
        ----------
        value:
            value that should be added to the series
        in_column:
            column to apply transform
        inplace:
            if True, apply add constant transformation inplace to in_column, if False, add transformed column to dataset
        out_column:
            name of added column. If not given, use '{in_column}_{self.__repr__()}'
        """
        self.value = value
        self.in_column = in_column
        self.inplace = inplace

        if inplace:
            out_column_result = in_column
        elif out_column:
            out_column_result = out_column
        else:
            out_column_result = f"{in_column}_{self.__repr__()}"

        super().__init__(
            transform=_OneSegmentAddConstTransform(
                value=value,
                in_column=in_column,
                inplace=inplace,
                out_column=out_column_result
            )
        )


__all__ = ["AddConstTransform"]
