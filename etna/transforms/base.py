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


class NewTransform(ABC, BaseMixin):
    """Base class to create any transforms to apply to data."""

    def __init__(self, in_column: Union[Literal["all"], List[str], str] = "target"):
        self.in_column = in_column

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.

        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return []

    @property
    def required_features(self) -> Union[Literal["all"], List[str]]:
        """Get the list of required features."""
        required_features = self.in_column
        if isinstance(required_features, list):
            return required_features
        elif isinstance(required_features, str):
            if required_features == "all":
                return "all"
            return [required_features]
        else:
            raise ValueError("in_column attribute is in incorrect format!")

    def _update_dataset(self, ts: TSDataset, df: pd.DataFrame, df_transformed: pd.DataFrame) -> TSDataset:
        """Update TSDataset based on the difference between dfs."""
        columns_before = set(df.columns.get_level_values("feature"))
        columns_after = set(df_transformed.columns.get_level_values("feature"))

        # Transforms now can only remove or only add/update columns
        removed_features = list(columns_before - columns_after)
        if len(removed_features) != 0:
            ts.remove_features(features=removed_features)
        else:
            new_regressors = self.get_regressors_info()
            ts.update_columns_from_pandas(df=df_transformed, regressors=new_regressors)
        return ts

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> "NewTransform":
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

    def fit(self, ts: TSDataset) -> "NewTransform":
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
        df = ts.to_pandas(flatten=False, features=self.required_features)
        self._fit(df=df)
        return self

    @abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe.

        Should be implemented by user

        Parameters
        ----------
        df:
            Dataframe in etna wide format.

        Returns
        -------
        :
            Transformed Dataframe in etna wide format.
        """
        pass

    def transform(self, ts: TSDataset) -> TSDataset:
        """Transform TSDataset inplace.

        Parameters
        ----------
        ts:
            Dataset to transform.

        Returns
        -------
        :
            Transformed TSDataset.
        """
        df = ts.to_pandas(flatten=False, features=self.required_features)
        df_transformed = self._transform(df=df)
        ts = self._update_dataset(ts=ts, df=df, df_transformed=df_transformed)
        return ts

    def fit_transform(self, ts: TSDataset) -> TSDataset:
        """Fit and transform TSDataset.

        May be reimplemented. But it is not recommended.

        Parameters
        ----------
        ts:
            TSDataset to transform.

        Returns
        -------
        :
            Transformed TSDataset.
        """
        return self.fit(ts=ts).transform(ts=ts)

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform dataframe.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.

        Returns
        -------
        :
            Dataframe in etna wide format after applying inverse transformation.
        """
        return df

    def inverse_transform(self, ts: TSDataset) -> TSDataset:
        """Inverse transform TSDataset.

        Should be reimplemented in the classes with reimplemented _inverse_transform method.

        Parameters
        ----------
        ts:
            TSDataset to be inverse transformed.

        Returns
        -------
        :
            TSDataset after applying inverse transformation.
        """
        return ts


class Transform(ABC, BaseMixin):
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
