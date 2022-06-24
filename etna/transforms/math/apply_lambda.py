import warnings
from typing import Callable
from typing import Optional

import pandas as pd

from etna.transforms import PerSegmentWrapper
from etna.transforms import Transform
from etna.transforms.utils import match_target_quantiles


class _OneSegmentLambdaTransform(Transform):
    """Instance of this class applies input function transformation to one segment data."""

    def __init__(
        self,
        in_column: str,
        inplace: bool,
        out_column: Optional[str],
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        inverse_transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        """
        Init OneSegmentLambdaTransform.

        Parameters
        ----------
        in_column:
            column to apply transform
        out_column:
            name of added column. If not given, use ``self.__repr__()``
        transform_func:
            function to transform data
        inverse_transform_func:
            inverse function of transform_func
        inplace:

            * if True, apply transformation inplace to in_column,

            * if False, add column and apply transformation to out_column

        Warnings
        --------
        throws if inplace=True and out_column is initialized, transformation will be applied inplace

        Raises
        ------
        Value error:
            if inplace=True and inverse_transform_func is not defined
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column
        self.transform_func = transform_func
        self.inverse_transform_func = inverse_transform_func

    def fit(self, df: pd.DataFrame) -> "Transform":
        """Fit preprocess method, does nothing in OneSegmentLambdaTransform case.

        Returns
        -------
        :
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
        :
            transformed series
        """
        result_df = df.copy()
        result_df[self.out_column] = self.transform_func(result_df[self.in_column])
        return result_df

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
            result_df[self.in_column] = self.inverse_transform_func(result_df[self.out_column])
            if self.in_column == "target":
                quantiles = match_target_quantiles(set(result_df.columns))
                for quantile_column_nm in quantiles:
                    result_df[quantile_column_nm] = self.inverse_transform_func(result_df[quantile_column_nm])
        return result_df


class LambdaTransform(PerSegmentWrapper):
    """LambdaTransform applies input function for given series."""

    def __init__(
        self,
        in_column: str,
        transform_func: Callable[[pd.DataFrame], pd.DataFrame],
        inplace: bool = True,
        out_column: Optional[str] = None,
        inverse_transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ):
        """Init LogTransform.

        Parameters
        ----------
        in_column:
            column to apply transform
        out_column:
            name of added column. If not given, use ``self.__repr__()``
        transform_func:
            function to transform data
        inverse_transform_func:
            inverse function of transform_func
        inplace:

            * if True, apply transformation inplace to in_column,

            * if False, add column and apply transformation to out_column

        Warnings
        --------
        throws if inplace=True and out_column is initialized, transformation will be applied inplace

        Raises
        ------
        Value error:
            if inplace=True and inverse_transform_func is not defined
        """
        self.in_column = in_column
        self.inplace = inplace
        self.out_column = out_column

        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")

        if self.inplace and inverse_transform_func is None:
            raise ValueError("inverse_transform_func must be defined, when inplace=True")

        if self.inplace:
            result_out_column = self.in_column
        elif out_column is not None:
            result_out_column = out_column
        else:
            result_out_column = self.__repr__()

        super().__init__(
            transform=_OneSegmentLambdaTransform(
                in_column=in_column,
                inplace=inplace,
                out_column=result_out_column,
                transform_func=transform_func,
                inverse_transform_func=inverse_transform_func,
            )
        )
