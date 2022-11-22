from abc import abstractmethod
from copy import deepcopy

import pandas as pd

from etna.core import AbstractSaveable
from etna.core import BaseMixin
from etna.core import SaveMixin


class FutureMixin:
    """Mixin for transforms that can convert non-regressor column to a regressor one."""


class Transform(SaveMixin, AbstractSaveable, BaseMixin):
    """Base class to create any transforms to apply to data."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "Transform":
        """Fit feature model.

        Should be implemented by user.

        Parameters
        ----------
        df

        Returns
        -------
        :
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe.

        Should be implemented by user

        Parameters
        ----------
        df

        Returns
        -------
        :
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        May be reimplemented. But it is not recommended.

        Parameters
        ----------
        df

        Returns
        -------
        :
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms dataframe.

        Parameters
        ----------
        df

        Returns
        -------
        :
        """
        return df


class PerSegmentWrapper(Transform):
    """Class to apply transform in per segment manner."""

    def __init__(self, transform):
        self._base_transform = transform
        self.segment_transforms = {}
        self.segments = None

    def fit(self, df: pd.DataFrame) -> "PerSegmentWrapper":
        """Fit transform on each segment."""
        self.segments = df.columns.get_level_values(0).unique()
        for segment in self.segments:
            self.segment_transforms[segment] = deepcopy(self._base_transform)
            self.segment_transforms[segment].fit(df[segment])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to each segment separately."""
        results = []
        for key, value in self.segment_transforms.items():
            seg_df = value.transform(df[key])

            _idx = seg_df.columns.to_frame()
            _idx.insert(0, "segment", key)
            seg_df.columns = pd.MultiIndex.from_frame(_idx)

            results.append(seg_df)
        df = pd.concat(results, axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse_transform to each segment."""
        results = []
        for key, value in self.segment_transforms.items():
            seg_df = value.inverse_transform(df[key])

            _idx = seg_df.columns.to_frame()
            _idx.insert(0, "segment", key)
            seg_df.columns = pd.MultiIndex.from_frame(_idx)

            results.append(seg_df)
        df = pd.concat(results, axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return df
