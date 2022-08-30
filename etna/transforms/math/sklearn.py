import warnings
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from etna.core import StringEnumWithRepr
from etna.datasets import set_columns_wide
from etna.transforms.base import Transform
from etna.transforms.utils import match_target_quantiles


class TransformMode(StringEnumWithRepr):
    """Enum for different metric aggregation modes."""

    macro = "macro"
    per_segment = "per-segment"


class SklearnTransform(Transform):
    """Base class for different sklearn transforms."""

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]],
        out_column: Optional[str],
        transformer: TransformerMixin,
        inplace: bool = True,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """
        Init SklearnTransform.

        Parameters
        ----------
        in_column:
            columns to be transformed, if None - all columns will be transformed.
        transformer:
            :py:class:`sklearn.base.TransformerMixin` instance.
        inplace:
            features are changed by transformed.
        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        mode:
            "macro" or "per-segment", way to transform features over segments.

            * If "macro", transforms features globally, gluing the corresponding ones for all segments.

            * If "per-segment", transforms features for each segment separately.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        if inplace and (out_column is not None):
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")

        self.transformer = transformer

        if isinstance(in_column, str):
            in_column = [in_column]
        self.in_column = in_column if in_column is None else sorted(in_column)

        self.inplace = inplace
        self.mode = TransformMode(mode)
        self.out_column = out_column

        self.out_columns: Optional[List[str]] = None

    def _get_column_name(self, in_column: str) -> str:
        if self.out_column is None:
            new_transform = deepcopy(self)
            new_transform.in_column = [in_column]
            return repr(new_transform)
        else:
            return f"{self.out_column}_{in_column}"

    def fit(self, df: pd.DataFrame) -> "SklearnTransform":
        """
        Fit transformer with data from df.

        Parameters
        ----------
        df:
            DataFrame to fit transformer.

        Returns
        -------
        :
        """
        df = df.sort_index(axis=1)

        if self.in_column is None:
            self.in_column = sorted(set(df.columns.get_level_values("feature")))

        if self.inplace:
            self.out_columns = self.in_column
        else:
            self.out_columns = [self._get_column_name(column) for column in self.in_column]

        if self.mode == TransformMode.per_segment:
            x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
        elif self.mode == TransformMode.macro:
            x = self._reshape(df)
        else:
            raise ValueError(f"'{self.mode}' is not a valid TransformMode.")

        self.transformer.fit(X=x)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform given data with fitted transformer.

        Parameters
        ----------
        df:
            DataFrame to transform with transformer.

        Returns
        -------
        :
            transformed DataFrame.
        """
        df = df.sort_index(axis=1)
        segments = sorted(set(df.columns.get_level_values("segment")))

        if self.mode == TransformMode.per_segment:
            x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
            transformed = self.transformer.transform(X=x)

        elif self.mode == TransformMode.macro:
            x = self._reshape(df)
            transformed = self.transformer.transform(X=x)
            transformed = self._inverse_reshape(df, transformed)
        else:
            raise ValueError(f"'{self.mode}' is not a valid TransformMode.")
        if self.inplace:
            df.loc[:, pd.IndexSlice[:, self.in_column]] = transformed
        else:
            transformed_features = pd.DataFrame(
                transformed, columns=df.loc[:, pd.IndexSlice[:, self.in_column]].columns, index=df.index
            ).sort_index(axis=1)
            transformed_features.columns = pd.MultiIndex.from_product([segments, self.out_columns])
            df = pd.concat((df, transformed_features), axis=1)
            df = df.sort_index(axis=1)

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation to DataFrame.

        Parameters
        ----------
        df:
            DataFrame to apply inverse transform.

        Returns
        -------
        :
            transformed DataFrame.
        """
        df = df.sort_index(axis=1)
        if self.in_column is None:
            raise ValueError("Transform is not fitted yet.")

        if "target" in self.in_column:
            quantiles = match_target_quantiles(set(df.columns.get_level_values("feature")))
        else:
            quantiles = set()

        if self.inplace:
            quantiles_arrays: Dict[str, pd.DataFrame] = dict()

            if self.mode == TransformMode.per_segment:
                x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
                transformed = self.transformer.inverse_transform(X=x)

                # quantiles inverse transformation
                for quantile_column_nm in quantiles:
                    df_slice_copy = df.loc[:, pd.IndexSlice[:, self.in_column]].copy()
                    df_slice_copy = set_columns_wide(
                        df_slice_copy, df, features_left=["target"], features_right=[quantile_column_nm]
                    )
                    transformed_quantile = self.transformer.inverse_transform(X=df_slice_copy)
                    df_slice_copy.loc[:, pd.IndexSlice[:, self.in_column]] = transformed_quantile
                    quantiles_arrays[quantile_column_nm] = df_slice_copy.loc[:, pd.IndexSlice[:, "target"]].rename(
                        columns={"target": quantile_column_nm}
                    )

            elif self.mode == TransformMode.macro:
                x = self._reshape(df)
                transformed = self.transformer.inverse_transform(X=x)
                transformed = self._inverse_reshape(df, transformed)

                # quantiles inverse transformation
                for quantile_column_nm in quantiles:
                    df_slice_copy = df.loc[:, pd.IndexSlice[:, self.in_column]].copy()
                    df_slice_copy = set_columns_wide(
                        df_slice_copy, df, features_left=["target"], features_right=[quantile_column_nm]
                    )
                    df_slice_copy_reshaped_array = self._reshape(df_slice_copy)
                    transformed_quantile = self.transformer.inverse_transform(X=df_slice_copy_reshaped_array)
                    inverse_reshaped_quantile = self._inverse_reshape(df_slice_copy, transformed_quantile)
                    df_slice_copy.loc[:, pd.IndexSlice[:, self.in_column]] = inverse_reshaped_quantile
                    quantiles_arrays[quantile_column_nm] = df_slice_copy.loc[:, pd.IndexSlice[:, "target"]].rename(
                        columns={"target": quantile_column_nm}
                    )

            else:
                raise ValueError(f"'{self.mode}' is not a valid TransformMode.")
            df.loc[:, pd.IndexSlice[:, self.in_column]] = transformed

            for quantile_column_nm in quantiles:
                df.loc[:, pd.IndexSlice[:, quantile_column_nm]] = quantiles_arrays[quantile_column_nm].values
        return df

    def _reshape(self, df: pd.DataFrame) -> np.ndarray:
        segments = sorted(set(df.columns.get_level_values("segment")))
        x = df.loc[:, pd.IndexSlice[:, self.in_column]]
        x = pd.concat([x[segment] for segment in segments]).values
        return x

    def _inverse_reshape(self, df: pd.DataFrame, transformed: np.ndarray) -> np.ndarray:
        time_period_len = len(df)
        n_segments = len(set(df.columns.get_level_values("segment")))
        transformed = np.concatenate(
            [transformed[i * time_period_len : (i + 1) * time_period_len, :] for i in range(n_segments)], axis=1
        )
        return transformed
