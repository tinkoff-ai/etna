import warnings
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from etna.core import StringEnumWithRepr
from etna.datasets import TSDataset
from etna.datasets import set_columns_wide
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import check_new_segments
from etna.transforms.utils import match_target_quantiles


class TransformMode(StringEnumWithRepr):
    """Enum for different metric aggregation modes."""

    macro = "macro"
    per_segment = "per-segment"


class SklearnTransform(ReversibleTransform):
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
        if isinstance(in_column, str):
            in_column = [in_column]
        required_features = sorted(in_column) if in_column is not None else "all"
        super().__init__(required_features=required_features)  # type: ignore

        if inplace and (out_column is not None):
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")
        self.in_column = in_column if in_column is None else sorted(in_column)
        self.transformer = transformer

        self.inplace = inplace
        self.mode = TransformMode(mode)
        self.out_column = out_column

        self.out_columns: Optional[List[str]] = None
        self.out_column_regressors: Optional[List[str]] = None
        self._fit_segments: Optional[List[str]] = None

    def _get_column_name(self, in_column: str) -> str:
        if self.out_column is None:
            new_transform = deepcopy(self)
            new_transform.in_column = [in_column]
            return repr(new_transform)
        else:
            return f"{self.out_column}_{in_column}"

    def _fit(self, df: pd.DataFrame) -> "SklearnTransform":
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

        self._fit_segments = df.columns.get_level_values("segment").unique().tolist()
        if self.mode == TransformMode.per_segment:
            x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
        elif self.mode == TransformMode.macro:
            x = self._preprocess_macro(df)
        else:
            raise ValueError(f"'{self.mode}' is not a valid TransformMode.")

        self.transformer.fit(X=x)
        return self

    def fit(self, ts: TSDataset) -> "SklearnTransform":
        """Fit the transform."""
        super().fit(ts)
        if self.in_column is None:
            raise ValueError("Something went wrong during the fit, cat not recognize in_column!")
        self.out_column_regressors = [
            self._get_column_name(in_column) for in_column in self.in_column if in_column in ts.regressors
        ]
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self._fit_segments is None:
            raise ValueError("The transform isn't fitted!")
        else:
            self.in_column = cast(List[str], self.in_column)

        df = df.sort_index(axis=1)
        transformed = self._make_transform(df)

        if self.inplace:
            df.loc[:, pd.IndexSlice[:, self.in_column]] = transformed
        else:
            segments = sorted(set(df.columns.get_level_values("segment")))
            transformed_features = pd.DataFrame(
                transformed, columns=df.loc[:, pd.IndexSlice[:, self.in_column]].columns, index=df.index
            ).sort_index(axis=1)
            transformed_features.columns = pd.MultiIndex.from_product([segments, self.out_columns])
            df = pd.concat((df, transformed_features), axis=1)
            df = df.sort_index(axis=1)

        return df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self._fit_segments is None:
            raise ValueError("The transform isn't fitted!")
        else:
            self.in_column = cast(List[str], self.in_column)

        df = df.sort_index(axis=1)

        if "target" in self.in_column:
            quantiles = match_target_quantiles(set(df.columns.get_level_values("feature")))
        else:
            quantiles = set()

        if self.inplace:
            quantiles_arrays: Dict[str, pd.DataFrame] = dict()
            transformed = self._make_inverse_transform(df)

            # quantiles inverse transformation
            for quantile_column_nm in quantiles:
                df_slice_copy = df.loc[:, pd.IndexSlice[:, self.in_column]].copy()
                df_slice_copy = set_columns_wide(
                    df_slice_copy, df, features_left=["target"], features_right=[quantile_column_nm]
                )
                transformed_quantile = self._make_inverse_transform(df_slice_copy)
                df_slice_copy.loc[:, pd.IndexSlice[:, self.in_column]] = transformed_quantile
                quantiles_arrays[quantile_column_nm] = df_slice_copy.loc[:, pd.IndexSlice[:, "target"]].rename(
                    columns={"target": quantile_column_nm}
                )

            df.loc[:, pd.IndexSlice[:, self.in_column]] = transformed
            for quantile_column_nm in quantiles:
                df.loc[:, pd.IndexSlice[:, quantile_column_nm]] = quantiles_arrays[quantile_column_nm].values

        return df

    def _preprocess_macro(self, df: pd.DataFrame) -> np.ndarray:
        segments = sorted(set(df.columns.get_level_values("segment")))
        x = df.loc[:, pd.IndexSlice[:, self.in_column]]
        x = pd.concat([x[segment] for segment in segments]).values
        return x

    def _postprocess_macro(self, df: pd.DataFrame, transformed: np.ndarray) -> np.ndarray:
        time_period_len = len(df)
        n_segments = len(set(df.columns.get_level_values("segment")))
        transformed = np.concatenate(
            [transformed[i * time_period_len : (i + 1) * time_period_len, :] for i in range(n_segments)], axis=1
        )
        return transformed

    def _preprocess_per_segment(self, df: pd.DataFrame) -> np.ndarray:
        self._fit_segments = cast(List[str], self._fit_segments)
        transform_segments = df.columns.get_level_values("segment").unique().tolist()
        check_new_segments(transform_segments=transform_segments, fit_segments=self._fit_segments)

        df = df.loc[:, pd.IndexSlice[:, self.in_column]]
        to_add_segments = set(self._fit_segments) - set(transform_segments)
        df_to_add = pd.DataFrame(index=df.index, columns=pd.MultiIndex.from_product([to_add_segments, self.in_column]))
        df = pd.concat([df, df_to_add], axis=1)
        df = df.sort_index(axis=1)
        return df.values

    def _postprocess_per_segment(self, df: pd.DataFrame, transformed: np.ndarray) -> np.ndarray:
        self._fit_segments = cast(List[str], self._fit_segments)
        self.in_column = cast(List[str], self.in_column)
        num_features = len(self.in_column)
        transform_segments = set(df.columns.get_level_values("segment"))
        select_segments = [segment in transform_segments for segment in self._fit_segments]
        # make a mask for columns to select
        select_columns = np.repeat(select_segments, num_features)
        result = transformed[:, select_columns]
        return result

    def _make_transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mode == TransformMode.per_segment:
            x = self._preprocess_per_segment(df)
            transformed = self.transformer.transform(X=x)
            transformed = self._postprocess_per_segment(df, transformed)
        elif self.mode == TransformMode.macro:
            x = self._preprocess_macro(df)
            transformed = self.transformer.transform(X=x)
            transformed = self._postprocess_macro(df, transformed)
        else:
            raise ValueError(f"'{self.mode}' is not a valid TransformMode.")
        return transformed

    def _make_inverse_transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mode == TransformMode.per_segment:
            x = self._preprocess_per_segment(df)
            transformed = self.transformer.inverse_transform(X=x)
            transformed = self._postprocess_per_segment(df, transformed)
        elif self.mode == TransformMode.macro:
            x = self._preprocess_macro(df)
            transformed = self.transformer.inverse_transform(X=x)
            transformed = self._postprocess_macro(df, transformed)
        else:
            raise ValueError(f"'{self.mode}' is not a valid TransformMode.")
        return transformed

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.out_column_regressors is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        if self.inplace:
            return []
        return self.out_column_regressors

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes ``mode`` parameter. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "mode": CategoricalDistribution(["per-segment", "macro"]),
        }
