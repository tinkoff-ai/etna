from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from sklearn.base import TransformerMixin

from etna.transforms.base import Transform


class SklearnTransform(Transform):
    """Base class for different sklearn transforms.
    TODO: current transforms make per column transforamtion, next per feature transforms should be added
    """

    def __init__(
        self, transformer: TransformerMixin, in_column: Optional[Union[str, List[str]]] = None, inplace: bool = True
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
        """
        self.transformer = transformer
        if isinstance(in_column, str):
            in_column = [in_column]
        self.in_column: Optional[List[str]] = in_column if in_column is None else sorted(in_column)
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
        if self.in_column is None:
            self.in_column = sorted(set(df.columns.get_level_values("feature")))
        x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
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
        x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
        transformed = self.transformer.transform(X=x)
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
            x = df.loc[:, pd.IndexSlice[:, self.in_column]].values
            transformed = self.transformer.inverse_transform(X=x)
            df.loc[:, pd.IndexSlice[:, self.in_column]] = transformed
        return df

    def __str__(self) -> str:
        return self.__class__.__name__.lower()
