from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils._encode import _check_unknown
from sklearn.utils._encode import _encode

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    new_value = "new_value"
    mean = "mean"
    none = "none"


class _LabelEncoder(preprocessing.LabelEncoder):
    def transform(self, y: pd.Series, strategy: str):
        diff = _check_unknown(y, known_values=self.classes_)

        index = np.where(np.isin(y, diff))[0]

        encoded = _encode(y, uniques=self.classes_, check_unknown=False).astype(float)

        if strategy == ImputerMode.none:
            filling_value = None
        elif strategy == ImputerMode.new_value:
            filling_value = -1
        elif strategy == ImputerMode.mean:
            filling_value = np.mean(encoded[~np.isin(y, diff)])
        else:
            raise ValueError(f"The strategy '{strategy}' doesn't exist")

        encoded[index] = filling_value
        return encoded


class _OneSegmentLabelEncoderTransform(Transform):
    """Replace the values in the column with the Label encoding."""

    def __init__(self, in_column: str, out_column: str, strategy: str):
        """
        Create instance of _OneSegmentLabelEncoderTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column.
        strategy:
            filling encoding in not fitted values:
            - If "new_value", then replace missing dates with '-1'
            - If "mean", then replace missing dates using the mean in encoded column
            - If "none", then replace missing dates with None
        """
        self.in_column = in_column
        self.out_column = out_column
        self.strategy = strategy
        self.le = _LabelEncoder()

    def fit(self, df: pd.DataFrame) -> "_OneSegmentLabelEncoderTransform":
        """
        Fit Label encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit the transform.
        Returns
        -------
        self
        """
        self.le.fit(df[self.in_column])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the `in_column` by fitted Label encoder.

        Parameters
        ----------
        df
            dataframe with data to transform.
        Returns
        -------
        result dataframe
        """
        result_df = df.copy()
        result_df[self.out_column] = self.le.transform(df[self.in_column], self.strategy)
        result_df[self.out_column] = result_df[self.out_column].astype("category")
        return result_df


class LabelEncoderTransform(PerSegmentWrapper):
    """Encode categorical feature with value between 0 and n_classes-1."""

    def __init__(self, in_column: str, out_column: Optional[str] = None, strategy: str = ImputerMode.mean):
        """
        Init LabelEncoderTransform.

        Parameters
        ----------
        in_column:
            name of column to be transformed
        out_column:
            name of added column. If not given, use `self.__repr__()` or `regressor_{self.__repr__()}` if it is a regressor
        strategy:
            filling encoding in not fitted values:
            - If "new_value", then replace missing values with '-1'
            - If "mean", then replace missing values using the mean in encoded column
            - If "none", then replace missing values with None
        """
        self.in_column = in_column
        self.strategy = strategy
        self.out_column = out_column
        super().__init__(
            transform=_OneSegmentLabelEncoderTransform(
                in_column=self.in_column, out_column=self._get_column_name(), strategy=self.strategy
            )
        )

    def _get_column_name(self) -> str:
        """Get the `out_column` depending on the transform's parameters."""
        if self.out_column:
            return self.out_column
        if self.in_column.startswith("regressor"):
            return f"regressor_{self.__repr__()}"
        return self.__repr__()


class _OneSegmentOneHotEncoderTransform(Transform):
    """Create one-hot encoding columns."""

    def __init__(self, in_column: str, out_column: str):
        """
        Create instance of _OneSegmentOneHotEncoderTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column
        """
        self.in_column = in_column
        self.out_column = out_column
        self.ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, df: pd.DataFrame) -> "_OneSegmentOneHotEncoderTransform":
        """
        Fit One Hot encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit the transform.
        Returns
        -------
        self
        """
        self.ohe.fit(np.array(df[self.in_column]).reshape(-1, 1))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the `in_column` by fitted One Hot encoder.

        Parameters
        ----------
        df
            dataframe with data to transform.
        Returns
        -------
        result dataframe
        """
        result_df = df.copy()
        result_df[[self.out_column + "_" + str(i) for i in range(len(self.ohe.categories_[0]))]] = self.ohe.transform(
            np.array(df[self.in_column]).reshape(-1, 1)
        )
        result_df[[self.out_column + "_" + str(i) for i in range(len(self.ohe.categories_[0]))]] = result_df[
            [self.out_column + "_" + str(i) for i in range(len(self.ohe.categories_[0]))]
        ].astype("category")
        return result_df


class OneHotEncoderTransform(PerSegmentWrapper):
    """Encode categorical feature as a one-hot numeric features.

    If unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros.

    """

    def __init__(self, in_column: str, out_column: Optional[str] = None):
        """
        Init OneHotEncoderTransform.

        Parameters
        ----------
        in_column:
            name of column to be encoded
        out_column:
            prefix of names of added columns. If not given, use `self.__repr__()` or `regressor_{self.__repr__()}` if it is a regressor
        """
        self.in_column = in_column
        self.out_column = out_column
        super().__init__(
            transform=_OneSegmentOneHotEncoderTransform(in_column=self.in_column, out_column=self._get_column_name())
        )

    def _get_column_name(self) -> str:
        """Get the `out_column` depending on the transform's parameters."""
        if self.out_column:
            return self.out_column
        if self.in_column.startswith("regressor"):
            return f"regressor_{self.__repr__()}"
        return self.__repr__()
