import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils._encode import _check_unknown
from sklearn.utils._encode import _encode
from sklearn.utils._encode import _unique

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class LE(preprocessing.LabelEncoder):
    def transform(self, y: pd.Series, strategy):
        diff = _check_unknown(y, known_values=self.classes_)

        index = np.where(np.isin(y, diff))[0]

        encoded = _encode(y, uniques=self.classes_, check_unknown=False).astype(float)

        if strategy == "None":
            filling_value = None
        elif strategy == "new_value":
            filling_value = -1
        elif strategy == "mean":
            filling_value = np.mean(encoded[~np.isin(y, diff)])
        else:
            raise ValueError(f"There are no '{strategy}' strategy exists")

        encoded[index] = filling_value
        return encoded


class _OneSegmentLabelEncoderTransform(Transform):
    """Replace the values in the column with the Label encoding"""

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
        self.le = LE()

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
        return result_df


class LabelEncoderTransform(PerSegmentWrapper):
    def __init__(self, in_column: str, inplace: bool = True, out_column: Optional[str] = None, strategy: str = "mean"):
        """
        Init LabelEncoderTransform.

        Parameters
        ----------
        in_column:
            name of column to be resampled
        inplace:
            if True, apply resampling inplace to in_column, if False, add transformed column to dataset
        out_column:
            name of added column. If not given, use `self.__repr__()` or `regressor_{self.__repr__()}` if it is a regressor
        strategy:
            filling encoding in not fitted values:
            - If "new_value", then replace missing dates with '-1'
            - If "mean", then replace missing dates using the mean in encoded column
            - If "none", then replace missing dates with None
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = self._get_out_column(out_column)
        self.strategy = strategy
        super().__init__(transform=_OneSegmentLabelEncoderTransform(self.in_column, self.out_column, self.strategy))

    def _get_out_column(self, out_column: Optional[str]) -> str:
        """Get the `out_column` depending on the transform's parameters."""
        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")
        if self.inplace:
            return self.in_column
        if out_column:
            return out_column
        if self.in_column.startswith("regressor"):
            return "regressor_" + self.__repr__()
        return self.__repr__()


###############################


class LB(preprocessing.LabelBinarizer):
    def transform(self, y: pd.Series, strategy):
        diff = _check_unknown(y, known_values=self.classes_)

        index = np.where(np.isin(y, diff))[0]

        encoded = _encode(y, uniques=self.classes_, check_unknown=False).astype(float)

        if strategy == "None":
            filling_value = None
        elif strategy == "new_value":
            filling_value = -1
        elif strategy == "mean":
            filling_value = np.mean(encoded[~np.isin(y, diff)])
        else:
            raise ValueError(f"There are no '{strategy}' strategy exists")

        encoded[index] = filling_value
        return encoded


class _OneSegmentLabelBinarizerTransform(Transform):
    """Create one-hot encoding columns"""

    def __init__(self, in_column: str, out_column: str):
        """
        Create instance of _OneSegmentLabelBinarizerTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column
        """
        self.in_column = in_column
        self.out_column = out_column
        self.lb = preprocessing.LabelBinarizer()

    def fit(self, df: pd.DataFrame) -> "_OneSegmentLabelBinarizerTransform":
        """
        Fit Label Binarizer encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit the transform.
        Returns
        -------
        self
        """
        self.lb.fit(df[self.in_column])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the `in_column` by fitted Label Binarize encoder.

        Parameters
        ----------
        df
            dataframe with data to transform.
        Returns
        -------
        result dataframe
        """
        result_df = df.copy()
        result_df[[self.out_column + "_" + str(i) for i in range(len(self.lb.classes_))]] = self.lb.transform(
            df[self.in_column]
        )
        return result_df


class LabelBinarizerTransform(PerSegmentWrapper):
    def __init__(self, in_column: str, inplace: bool = True, out_column: Optional[str] = None):
        """
        Init LabelBinarizerTransform.

        Parameters
        ----------
        in_column:
            name of column to be resampled
        inplace:
            if True, apply resampling inplace to in_column, if False, add transformed column to dataset
        out_column:
            prefix of names of added column. If not given, use `self.__repr__()` or `regressor_{self.__repr__()}` if it is a regressor
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = self._get_out_column(out_column)
        super().__init__(transform=_OneSegmentLabelBinarizerTransform(self.in_column, self.out_column))

    def _get_out_column(self, out_column: Optional[str]) -> str:
        """Get the `out_column` depending on the transform's parameters."""
        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")
        if self.inplace:
            return self.in_column
        if out_column:
            return out_column
        if self.in_column.startswith("regressor"):
            return "regressor_" + self.__repr__()
        return self.__repr__()
