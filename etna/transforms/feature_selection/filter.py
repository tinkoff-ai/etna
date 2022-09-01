from typing import Optional
from typing import Sequence

import pandas as pd

from etna.transforms.base import Transform


class FilterFeaturesTransform(Transform):
    """Filters features in each segment of the dataframe."""

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        return_features: bool = False,
    ):
        """Create instance of FilterFeaturesTransform.

        Parameters
        ----------
        include:
            list of columns to pass through filter
        exclude:
            list of columns to not pass through
        return_features:
            indicates whether to return features or not.
        Raises
        ------
        ValueError:
            if both options set or non of them
        """
        self.include: Optional[Sequence[str]] = None
        self.exclude: Optional[Sequence[str]] = None
        self.return_features: bool = return_features
        self._df_removed: Optional[pd.DataFrame] = None
        if include is not None and exclude is None:
            self.include = list(set(include))
        elif exclude is not None and include is None:
            self.exclude = list(set(exclude))
        else:
            raise ValueError("There should be exactly one option set: include or exclude")

    def fit(self, df: pd.DataFrame) -> "FilterFeaturesTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: FilterFeaturesTransform
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter features according to include/exclude parameters.

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
        features = df.columns.get_level_values("feature")
        if self.include is not None:
            if not set(self.include).issubset(features):
                raise ValueError(f"Features {set(self.include) - set(features)} are not present in the dataset.")
            result = result.loc[:, pd.IndexSlice[:, self.include]]
        if self.exclude is not None and self.exclude:
            if not set(self.exclude).issubset(features):
                raise ValueError(f"Features {set(self.exclude) - set(features)} are not present in the dataset.")
            result = result.drop(columns=self.exclude, level="feature")
        if self.return_features:
            self._df_removed = df.drop(result.columns, axis=1)

        result = result.sort_index(axis=1)
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
