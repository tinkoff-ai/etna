import warnings
from typing import Callable
from typing import Optional

import pandas as pd

from etna.datasets import set_columns_wide
from etna.transforms.base import Transform
from etna.transforms.utils import match_target_quantiles


class LambdaTransform(Transform):
    """``LambdaTransform`` applies input function for given series."""

    def __init__(
        self,
        in_column: str,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        inplace: bool = True,
        out_column: Optional[str] = None,
        inverse_transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        """Init ``LambdaTransform``.

        Parameters
        ----------
        in_column:
            column to apply transform
        out_column:
            name of added column. If not given, use ``self.__repr__()``
        transform_func:
            function to transform data
        inverse_transform_func:
            inverse function of ``transform_func``
        inplace:

            * if `True`, apply transformation inplace to ``in_column``,

            * if `False`, add column and apply transformation to ``out_column``

        Warnings
        --------
        throws if `inplace=True` and ``out_column`` is initialized, transformation will be applied inplace

        Raises
        ------
        Value error:
            if `inplace=True` and ``inverse_transform_func`` is not defined
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column
        self.transform_func = transform_func
        self.inverse_transform_func = inverse_transform_func

        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")

        if self.inplace and inverse_transform_func is None:
            raise ValueError("inverse_transform_func must be defined, when inplace=True")

        if self.inplace:
            self.change_column = self.in_column
        elif self.out_column is not None:
            self.change_column = self.out_column
        else:
            self.change_column = self.__repr__()

    def fit(self, df: pd.DataFrame) -> "LambdaTransform":
        """Fit preprocess method, does nothing in ``LambdaTransform`` case.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: ``LambdaTransform``
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply lambda transformation to series from df.

        Parameters
        ----------
        df:
            series to transform

        Returns
        -------
        :
            transformed series
        """
        result = df.copy()
        segments = sorted(set(df.columns.get_level_values("segment")))
        features = df.loc[:, pd.IndexSlice[:, self.in_column]].sort_index(axis=1)
        transformed_features = self.transform_func(features)
        if self.inplace:
            result = set_columns_wide(
                result, transformed_features, features_left=[self.in_column], features_right=[self.in_column]
            )
        else:
            transformed_features.columns = pd.MultiIndex.from_product([segments, [self.change_column]])
            result = pd.concat([result] + [transformed_features], axis=1)
            result = result.sort_index(axis=1)
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to the series from df.

        Parameters
        ----------
        df:
            series to transform

        Returns
        -------
        :
            transformed series
        """
        result_df = df.copy()
        if self.inverse_transform_func:
            features = df.loc[:, pd.IndexSlice[:, self.in_column]].sort_index(axis=1)
            transformed_features = self.inverse_transform_func(features)
            result_df = set_columns_wide(
                result_df, transformed_features, features_left=[self.in_column], features_right=[self.in_column]
            )
            if self.in_column == "target":
                segment_columns = result_df.columns.get_level_values("feature").tolist()
                quantiles = match_target_quantiles(set(segment_columns))
                for quantile_column_nm in quantiles:
                    features = df.loc[:, pd.IndexSlice[:, quantile_column_nm]].sort_index(axis=1)
                    transformed_features = self.inverse_transform_func(features)
                    result_df = set_columns_wide(
                        result_df,
                        transformed_features,
                        features_left=[quantile_column_nm],
                        features_right=[quantile_column_nm],
                    )
        return result_df
