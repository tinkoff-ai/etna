from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils._encode import _check_unknown
from sklearn.utils._encode import _encode

from etna.datasets import TSDataset
from etna.transforms.base import Transform


class ImputerMode(str, Enum):
    """Enum for different imputation strategy."""

    new_value = "new_value"
    mean = "mean"
    none = "none"


class _LabelEncoder(preprocessing.LabelEncoder):
    def transform(self, y: pd.Series, strategy: str):
        diff = _check_unknown(y, known_values=self.classes_)

        is_new_index = np.isin(y, diff)

        encoded = np.zeros(y.shape[0], dtype=float)
        encoded[~is_new_index] = _encode(y.iloc[~is_new_index], uniques=self.classes_, check_unknown=False).astype(
            float
        )

        if strategy == ImputerMode.none:
            filling_value = None
        elif strategy == ImputerMode.new_value:
            filling_value = -1
        elif strategy == ImputerMode.mean:
            filling_value = np.mean(encoded[~np.isin(y, diff)])
        else:
            raise ValueError(f"The strategy '{strategy}' doesn't exist")

        encoded[is_new_index] = filling_value
        return encoded


class LabelEncoderTransform(Transform):
    """Encode categorical feature with value between 0 and n_classes-1."""

    def __init__(self, in_column: str, out_column: Optional[str] = None, strategy: str = ImputerMode.mean):
        """
        Init LabelEncoderTransform.

        Parameters
        ----------
        in_column:
            Name of column to be transformed
        out_column:
            Name of added column. If not given, use ``self.__repr__()``
        strategy:
            Filling encoding in not fitted values:

            - If "new_value", then replace missing values with '-1'

            - If "mean", then replace missing values using the mean in encoded column

            - If "none", then replace missing values with None

        """
        self.in_column = in_column
        self.out_column = out_column
        self.strategy = strategy
        self.le = _LabelEncoder()

    def fit(self, df: pd.DataFrame) -> "LabelEncoderTransform":
        """
        Fit Label encoder.

        Parameters
        ----------
        df:
            Dataframe with data to fit the transform
        Returns
        -------
        :
            Fitted transform
        """
        y = TSDataset.to_flatten(df)[self.in_column]
        self.le.fit(y=y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the ``in_column`` by fitted Label encoder.

        Parameters
        ----------
        df
            Dataframe with data to transform

        Returns
        -------
        :
            Dataframe with column with encoded values
        """
        out_column = self._get_column_name()
        result_df = TSDataset.to_flatten(df)
        result_df[out_column] = self.le.transform(result_df[self.in_column], self.strategy)
        result_df[out_column] = result_df[out_column].astype("category")
        result_df = TSDataset.to_dataset(result_df)
        return result_df

    def _get_column_name(self) -> str:
        """Get the ``out_column`` depending on the transform's parameters."""
        if self.out_column:
            return self.out_column
        return self.__repr__()


class OneHotEncoderTransform(Transform):
    """Encode categorical feature as a one-hot numeric features.

    If unknown category is encountered during transform, the resulting one-hot
    encoded columns for this feature will be all zeros.
    """

    def __init__(self, in_column: str, out_column: Optional[str] = None):
        """
        Init OneHotEncoderTransform.

        Parameters
        ----------
        in_column:
            Name of column to be encoded
        out_column:
            Prefix of names of added columns. If not given, use ``self.__repr__()``
        """
        self.in_column = in_column
        self.out_column = out_column
        self.ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=int)

    def fit(self, df: pd.DataFrame) -> "OneHotEncoderTransform":
        """
        Fit One Hot encoder.

        Parameters
        ----------
        df:
            Dataframe with data to fit the transform
        Returns
        -------
        :
            Fitted transform
        """
        x = TSDataset.to_flatten(df)[[self.in_column]]
        self.ohe.fit(X=x)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the `in_column` by fitted One Hot encoder.

        Parameters
        ----------
        df
            Dataframe with data to transform

        Returns
        -------
        :
            Dataframe with column with encoded values
        """
        out_column = self._get_column_name()
        out_columns = [out_column + "_" + str(i) for i in range(len(self.ohe.categories_[0]))]
        result_df = TSDataset.to_flatten(df)
        x = result_df[[self.in_column]]
        result_df[out_columns] = self.ohe.transform(X=x)
        result_df[out_columns] = result_df[out_columns].astype("category")
        result_df = TSDataset.to_dataset(result_df)
        return result_df

    def _get_column_name(self) -> str:
        """Get the ``out_column`` depending on the transform's parameters."""
        if self.out_column:
            return self.out_column
        return self.__repr__()
