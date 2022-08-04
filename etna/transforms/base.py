from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import List
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna.core import BaseMixin
from etna.datasets import TSDataset


class FutureMixin:
    """Mixin for transforms that can convert non-regressor column to a regressor one."""


class DymmyInColumnMixin:
    """Mixin for transforms that has no explicit in_column."""

    in_column = "target"


class Transform(ABC, BaseMixin):
    """Base class to create any transforms to apply to data."""

    in_column: Union[Literal["all"], List[str], str] = "target"

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.

        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return []

    def _get_required_features(self) -> Union[Literal["all"], List[str]]:
        """Get the list of required features."""
        required_features = self.in_column
        if isinstance(required_features, str) and required_features != "all":
            required_features = [required_features]
        return required_features

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> "Transform":
        """Fit the transform.

        Should be implemented by user.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.

        Returns
        -------
        :
            The fitted transform instance.
        """
        pass

    def fit(self, ts: TSDataset) -> "Transform":
        """Fit the transform.

        Parameters
        ----------
        ts:
            Dataset to fit the transform on.

        Returns
        -------
        :
            The fitted transform instance.
        """
        features_to_use = self._get_required_features()
        df = ts.to_pandas(flatten=False, features=features_to_use)
        self._fit(df=df)
        return self

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
            self.segment_transforms[segment]._fit(df=df[segment])
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
