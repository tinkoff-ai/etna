from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import Transform


class LagTransform(Transform, FutureMixin):
    """Generates series of lags from given dataframe."""

    def __init__(self, in_column: str, lags: Union[List[int], int], out_column: Optional[str] = None):
        """Create instance of LagTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        lags:
            int value or list of values for lags computation; if int, generate range of lags from 1 to given value
        out_column:
            base for the name of created columns;

            * if set the final name is '{out_column}_{lag_number}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        Raises
        ------
        ValueError:
            if lags value contains non-positive values
        """
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError(f"{type(self).__name__} works only with positive lags values, {lags} given")
            self.lags = list(range(1, lags + 1))
        else:
            if any(lag_value < 1 for lag_value in lags):
                raise ValueError(f"{type(self).__name__} works only with positive lags values")
            self.lags = lags

        self.in_column = in_column
        self.out_column = out_column

    def _get_column_name(self, lag: int) -> str:
        if self.out_column is None:
            temp_transform = LagTransform(in_column=self.in_column, out_column=self.out_column, lags=[lag])
            return repr(temp_transform)
        else:
            return f"{self.out_column}_{lag}"

    def fit(self, df: pd.DataFrame) -> "LagTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: LagTransform
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lags to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        result = df.copy()
        segments = sorted(set(df.columns.get_level_values("segment")))
        all_transformed_features = []
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]
        for lag in self.lags:
            column_name = self._get_column_name(lag)
            transformed_features = features.shift(lag)
            transformed_features.columns = pd.MultiIndex.from_product([segments, [column_name]])
            all_transformed_features.append(transformed_features)
        result = pd.concat([result] + all_transformed_features, axis=1)
        result = result.sort_index(axis=1)
        return result
