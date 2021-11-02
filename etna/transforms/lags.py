from typing import List
from typing import Union

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentLagFeature(Transform):
    def __init__(self, in_column: str, lags: Union[List[int], int]):
        self.in_column = in_column
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError(f"{type(self).__name__} works only with positive lags values, {lags} given")
            self.lags = list(range(1, lags + 1))
        else:
            if any(lag_value < 1 for lag_value in lags):
                raise ValueError(f"{type(self).__name__} works only with positive lags values")
            self.lags = lags

        self.out_postfix = "_lag"
        self.out_prefix = "regressor_"

    def fit(self, *args) -> "_OneSegmentLagFeature":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for lag in self.lags:
            result[f"{self.out_prefix}{self.in_column}{self.out_postfix}_{lag}"] = df[self.in_column].shift(lag)
        return result


class LagTransform(PerSegmentWrapper):
    """Generates series of lags from given dataframe. Creates columns 'regressor_<column>_lag_<number>'."""

    def __init__(self, in_column: str, lags: Union[List[int], int]):
        """Create instance of LagTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        lags:
            int value or list of values for lags computation; if int, generate range of lags from 1 to given value

        Raises
        ------
        ValueError:
            if lags value contains non-positive values
        """
        self.in_column = in_column
        self.lags = lags
        super().__init__(transform=_OneSegmentLagFeature(in_column=self.in_column, lags=self.lags))
