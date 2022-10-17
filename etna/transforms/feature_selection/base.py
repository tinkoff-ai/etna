import warnings
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna.transforms import Transform


class BaseFeatureSelectionTransform(Transform):
    """Base class for feature selection transforms."""

    def __init__(self, features_to_use: Union[List[str], Literal["all"]] = "all", return_features: bool = False):
        self.features_to_use = features_to_use
        self.selected_features: List[str] = []
        self.return_features = return_features
        self._df_removed: Optional[pd.DataFrame] = None

    def _get_features_to_use(self, df: pd.DataFrame) -> List[str]:
        """Get list of features from the dataframe to perform the selection on."""
        features = set(df.columns.get_level_values("feature")) - {"target"}
        if self.features_to_use != "all":
            features = features.intersection(self.features_to_use)
            if sorted(features) != sorted(self.features_to_use):
                warnings.warn("Columns from feature_to_use which are out of dataframe columns will be dropped!")
        return sorted(features)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top_k features.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result: pd.DataFrame
            Dataframe with with only selected features
        """
        result = df.copy()
        rest_columns = set(df.columns.get_level_values("feature")) - set(self._get_features_to_use(df))
        selected_columns = sorted(self.selected_features + list(rest_columns))
        result = result.loc[:, pd.IndexSlice[:, selected_columns]]
        if self.return_features:
            self._df_removed = df.drop(result.columns, axis=1)
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transform to the data.

        Parameters
        ----------
        df:
            dataframe to apply inverse transformation

        Returns
        -------
        result: pd.DataFrame
            dataframe before transformation
        """
        return pd.concat([df, self._df_removed], axis=1)
