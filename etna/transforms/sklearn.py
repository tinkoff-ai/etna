from enum import Enum
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from etna.transforms.base import Transform


class TransformMode(str, Enum):
    """Enum for different metric aggregation modes."""

    macro = "macro"
    per_segment = "per-segment"


class SklearnTransform(Transform):
    """Base class for different sklearn transforms."""

    def __init__(
        self,
        transformer: TransformerMixin,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        mode: TransformMode = "per-segment",
    ):
        """
        Init SklearnTransform.

        Parameters
        ----------
        transformer:
            sklearn.base.TransformerMixin instance.
        in_column:
            columns to be transformed, if None - all columns will be scaled.
        inplace:
            features are changed by transformed.
        mode:
            "macro" or "per-segment", way to transform features over segments.
            If "macro", transforms features globally, gluing the corresponding ones for all segments.
            If "per-segment", transforms features for each segment separately.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        self.transformer = transformer
        if isinstance(in_column, str):
            in_column = [in_column]
        self.in_column: Optional[List[str]] = in_column if in_column is None else sorted(in_column)
        self.inplace = inplace
        self.mode = TransformMode(mode)

    def fit(self, df: pd.DataFrame) -> "SklearnTransform":
        """
        Fit transformer with data from df.

        Parameters
        ----------
        df:
            DataFrame to fit transformer.

        Returns
        -------
        self
        """
        if self.in_column is None:
            self.in_column = sorted(set(df.columns.get_level_values("feature")))
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
        transformed DataFrame.
        """
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
            )
            transformed_features.columns = pd.MultiIndex.from_tuples(
                [
                    (segment_name, f"{str(self)}_{feature_name}")
                    for segment_name, feature_name in transformed_features.columns
                ]
            )
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
        transformed DataFrame.
        """
        if self.inplace:
            if self.mode == TransformMode.per_segment:
                x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
                transformed = self.transformer.inverse_transform(X=x)

            elif self.mode == TransformMode.macro:
                x = self._reshape(df)
                transformed = self.transformer.inverse_transform(X=x)
                transformed = self._inverse_reshape(df, transformed)
            else:
                raise ValueError(f"'{self.mode}' is not a valid TransformMode.")
            df.loc[:, pd.IndexSlice[:, self.in_column]] = transformed
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

    def __str__(self) -> str:
        return self.__class__.__name__.lower()
