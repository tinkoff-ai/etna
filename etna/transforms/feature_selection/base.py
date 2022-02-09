import warnings
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

    def _get_features_to_use(self, df: pd.DataFrame) -> List[str]:
        """Get list of features from the dataframe to preform the selection on."""
        features = set(df.columns.get_level_values("feature"))
        if self.features_to_use != "all":
            features = features.intersection(self.features_to_use)
            if sorted(features) != sorted(self.features_to_use):
                warnings.warn("Columns from feature_to_use which are out of dataframe columns will be dropped!")
        return sorted(features)

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
