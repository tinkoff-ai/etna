from typing import List
from typing import Optional

import pandas as pd
from sklearn.base import TransformerMixin

from etna.transforms.base import Transform


class SklearnTransform(Transform):
    """Base class for different sklearn transforms.
    TODO: current transforms make per column transforamtion, next per feature transforms should be added
    """

    def __init__(self, transformer: TransformerMixin, in_columns: Optional[List[str]] = None, inplace: bool = True):
        """
        Init SklearnTransform.

        Parameters
        ----------
        transform:
            sklearn.base.TransformerMixin instance.
        in_columns:
            columns to be transformed, if None - all columns will be scaled.
        inplace:
            features are changed by transformed.
        """
        self.transformer = transformer
        self.in_columns: Optional[List[str]] = in_columns if in_columns is None else sorted(in_columns)
        self.inplace = inplace

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
        if self.in_columns is None:
            self.in_columns = sorted(set(df.columns.get_level_values("feature")))
        x = df.loc[:, pd.IndexSlice[:, self.in_columns]].values
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
        x = df.loc[:, pd.IndexSlice[:, self.in_columns]].values
        transformed = self.transformer.transform(X=x)
        if self.inplace:
            df.loc[:, pd.IndexSlice[:, self.in_columns]] = transformed
        else:
            transformed_features = pd.DataFrame(
                transformed, columns=df.loc[:, pd.IndexSlice[:, self.in_columns]].columns, index=df.index
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
            x = df.loc[:, pd.IndexSlice[:, self.in_columns]].values
            transformed = self.transformer.inverse_transform(X=x)
            df.loc[:, pd.IndexSlice[:, self.in_columns]] = transformed
        return df

    def __str__(self) -> str:
        return self.__class__.__name__.lower()
