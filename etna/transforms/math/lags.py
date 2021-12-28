from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentLagFeature(Transform):
    """Generates series of lags from given segment."""

    def __init__(self, in_column: str, lags: Union[List[int], int], out_column: Optional[str] = None):
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

    def fit(self, *args) -> "_OneSegmentLagFeature":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for lag in self.lags:
            result[f"{self.out_column}_{lag}"] = df[self.in_column].shift(lag)
        return result


class LagTransform(PerSegmentWrapper):
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
            name of added column. We get '{out_column}_{lag_number}'(don't forget to add regressor prefix if necessary).
            If not given, use 'regressor_{self.__repr__()}_{lag_number}'

        Raises
        ------
        ValueError:
            if lags value contains non-positive values
        """
        self.in_column = in_column
        self.lags = lags
        self.out_column = out_column

        super().__init__(
            transform=_OneSegmentLagFeature(
                in_column=self.in_column,
                lags=self.lags,
                out_column=out_column if out_column is not None else f"regressor_{self.__repr__()}",
            )
        )
