from abc import ABC
from typing import List
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna.transforms import Transform


class BaseFeatureSelectionTransform(Transform, ABC):
    """Base class for feature selection transforms."""

    def __init__(self, features_to_use: Union[List[str], Literal["all"]] = "all"):
        self.features_to_use = features_to_use
        self.selected_regressors: List[str] = []

    @staticmethod
    def _get_features_to_use(df: pd.DataFrame) -> List[str]:
        """Get list of regressors in the dataframe."""
        result = set()
        for column in df.columns.get_level_values("feature"):
            if column.startswith("regressor_"):
                result.add(column)
        return sorted(list(result))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top_k regressors.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result: pd.DataFrame
            Dataframe with with only selected regressors
        """
        result = df.copy()
        selected_columns = sorted(
            [
                column
                for column in df.columns.get_level_values("feature").unique()
                if not column.startswith("regressor_") or column in self.selected_regressors
            ]
        )
        result = result.loc[:, pd.IndexSlice[:, selected_columns]]
        return result
