from typing import Optional
from typing import Sequence

import pandas as pd

from etna.transforms.base import Transform


class FilterFeaturesTransform(Transform):
    """Filters features in each segment of the dataframe."""

    def __init__(self, include: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None):
        """Create instance of LagTransform.

        Parameters
        ----------
        include:
            list of columns to pass through filter
        exclude:
            list of columns to not pass through

        Raises
        ------
        ValueError:
            if both options set or non of them
        """
        if include is not None and exclude is None:
            self.include = list(set(include))
            self.exclude = None
        elif exclude is not None and include is None:
            self.include = None
            self.exclude = list(set(exclude))
        else:
            raise ValueError("There should be exactly one option set: include or exclude")

    def fit(self, df: pd.DataFrame) -> "FilterFeaturesTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data to fit label encoder.

        Returns
        -------
        self
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
        segments = sorted(set(df.columns.get_level_values("segment")))
        if self.include is not None:
            if not set(self.include).issubset(df.columns.get_level_values("feature")):
                raise ValueError("Some features in include are not present in the dataset")
            result = result.loc[:, (segments, self.include)]
        if self.exclude is not None and self.exclude:
            if not set(self.exclude).issubset(df.columns.get_level_values("feature")):
                raise ValueError("Some features in exclude are not present in the dataset")
            result = result.drop(columns=self.exclude, level="feature")
        return result
