import warnings
from typing import List
from typing import Optional

import pandas as pd

from etna.datasets import TSDataset
from etna.datasets import set_columns_wide
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import match_target_quantiles


class AddConstTransform(ReversibleTransform):
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

            * if True, apply add constant transformation inplace to in_column,

            * if False, add transformed column to dataset

        out_column:
            name of added column. If not given, use ``self.__repr__()``
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.value = value
        self.inplace = inplace
        self.out_column = out_column
        self.in_column_regressor: Optional[bool] = None

        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")

    def _get_column_name(self) -> str:
        if self.inplace:
            return self.in_column
        elif self.out_column:
            return self.out_column
        else:
            return self.__repr__()

    def _fit(self, df: pd.DataFrame) -> "AddConstTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: AddConstTransform
        """
        return self

    def fit(self, ts: TSDataset) -> "AddConstTransform":
        """Fit the transform."""
        self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply adding constant to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        result = df
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]
        transformed_features = features + self.value
        if self.inplace:
            result = set_columns_wide(
                result, transformed_features, features_left=[self.in_column], features_right=[self.in_column]
            )
        else:
            column_name = self._get_column_name()
            transformed_features.columns = pd.MultiIndex.from_product([segments, [column_name]])
            result = pd.concat((result, transformed_features), axis=1)
            result = result.sort_index(axis=1)
        return result

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.DataFrame
            transformed series
        """
        result = df
        if self.inplace:
            features = df.loc[:, pd.IndexSlice[:, self.in_column]]
            transformed_features = features - self.value
            result = set_columns_wide(
                result, transformed_features, features_left=[self.in_column], features_right=[self.in_column]
            )
            if self.in_column == "target":
                segment_columns = result.columns.get_level_values("feature").tolist()
                quantiles = match_target_quantiles(set(segment_columns))
                for quantile_column_nm in quantiles:
                    features = df.loc[:, pd.IndexSlice[:, quantile_column_nm]]
                    transformed_features = features - self.value
                    result = set_columns_wide(
                        result,
                        transformed_features,
                        features_left=[quantile_column_nm],
                        features_right=[quantile_column_nm],
                    )

        return result

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        return [self._get_column_name()] if self.in_column_regressor and not self.inplace else []


__all__ = ["AddConstTransform"]
