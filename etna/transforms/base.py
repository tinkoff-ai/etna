from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna.core import BaseMixin
from etna.core import SaveMixin
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.transforms.utils import match_target_quantiles


class FutureMixin:
    """Mixin for transforms that can convert non-regressor column to a regressor one."""


class Transform(SaveMixin, BaseMixin):
    """Base class to create any transforms to apply to data."""

    def __init__(self, required_features: Union[Literal["all"], List[str]]):
        self.required_features = required_features

    @abstractmethod
    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.

        Returns
        -------
        :
            List with regressors created by the transform.
        """
        pass

    def _update_dataset(self, ts: TSDataset, columns_before: Set[str], df_transformed: pd.DataFrame) -> TSDataset:
        """Update TSDataset based on the difference between dfs."""
        columns_after = set(df_transformed.columns.get_level_values("feature"))
        columns_to_update = list(set(columns_before) & set(columns_after))
        columns_to_add = list(set(columns_after) - set(columns_before))
        columns_to_remove = list(set(columns_before) - set(columns_after))

        if len(columns_to_remove) != 0:
            ts.drop_features(features=columns_to_remove, drop_from_exog=False)
        if len(columns_to_add) != 0:
            new_regressors = self.get_regressors_info()
            ts.add_columns_from_pandas(
                df_update=df_transformed.loc[pd.IndexSlice[:], pd.IndexSlice[:, columns_to_add]],
                update_exog=False,
                regressors=new_regressors,
            )
        if len(columns_to_update) != 0:
            ts.update_columns_from_pandas(
                df_update=df_transformed.loc[pd.IndexSlice[:], pd.IndexSlice[:, columns_to_update]]
            )
        return ts

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        """Fit the transform.

        Should be implemented by user.

        Parameters
        ----------
        df:
            Dataframe in etna wide format.
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
        columns_before = set(df.columns.get_level_values("feature"))
        df_transformed = self._transform(df=df)
        ts = self._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)
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

    @abstractmethod
    def inverse_transform(self, ts: TSDataset) -> TSDataset:
        """Inverse transform TSDataset.

        Should be reimplemented in the subclasses where necessary.

        Parameters
        ----------
        ts:
            TSDataset to be inverse transformed.

        Returns
        -------
        :
            TSDataset after applying inverse transformation.
        """
        pass

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get grid for tuning hyperparameters.

        This is default implementation with empty grid.

        Returns
        -------
        :
            Empty grid.
        """
        return {}


class IrreversibleTransform(Transform):
    """Base class to create irreversible transforms."""

    def __init__(self, required_features: Union[Literal["all"], List[str]]):
        super().__init__(required_features=required_features)

    def inverse_transform(self, ts: TSDataset) -> TSDataset:
        """Inverse transform TSDataset.

        Do nothing.

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


class ReversibleTransform(Transform):
    """Base class to create reversible transforms."""

    def __init__(self, required_features: Union[Literal["all"], List[str]]):
        super().__init__(required_features=required_features)

    @abstractmethod
    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform dataframe.

        Should be reimplemented in the subclasses where necessary.

        Parameters
        ----------
        df:
            Dataframe to be inverse transformed.

        Returns
        -------
        :
            Dataframe after applying inverse transformation.
        """
        pass

    def _get_inverse_transform_required_features(self, ts) -> Union[List[str], Literal["all"]]:
        """Add the target quantiles for the list with required features if necessary."""
        required_features = self.required_features
        if isinstance(required_features, list) and "target" in self.required_features:
            features = set(ts.columns.get_level_values("feature").tolist())
            required_features = list(set(required_features) | match_target_quantiles(features))
        return required_features

    def inverse_transform(self, ts: TSDataset) -> TSDataset:
        """Inverse transform TSDataset.

        Apply the _inverse_transform method.

        Parameters
        ----------
        ts:
            TSDataset to be inverse transformed.

        Returns
        -------
        :
            TSDataset after applying inverse transformation.
        """
        required_features = self._get_inverse_transform_required_features(ts)
        target_components_present = "target" in required_features and len(ts.target_components_names) > 0
        target_df = None
        if target_components_present:
            target_df = ts.to_pandas(flatten=False, features=["target"])

        df = ts.to_pandas(flatten=False, features=required_features)
        columns_before = set(df.columns.get_level_values("feature"))
        df_transformed = self._inverse_transform(df=df)
        ts = self._update_dataset(ts=ts, columns_before=columns_before, df_transformed=df_transformed)

        if target_components_present:
            ts._inverse_transform_target_components(
                target_components_df=ts.get_target_components(), target_df=target_df
            )

        return ts


class OneSegmentTransform(ABC, BaseMixin):
    """Base class to create one segment transforms to apply to data."""

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Fit the transform.

        Should be implemented by user.

        Parameters
        ----------
        df:
            Dataframe in etna long format.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe.

        Should be implemented by user

        Parameters
        ----------
        df:
            Dataframe in etna long format.

        Returns
        -------
        :
            Transformed Dataframe in etna long format.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform Dataframe.

        May be reimplemented. But it is not recommended.

        Parameters
        ----------
        df:
            Dataframe in etna long format to transform.

        Returns
        -------
        :
            Transformed Dataframe.
        """
        return self.fit(df=df).transform(df=df)

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform Dataframe.

        Should be reimplemented in the subclasses where necessary.

        Parameters
        ----------
        df:
            Dataframe in etna long format to be inverse transformed.

        Returns
        -------
        :
            Dataframe after applying inverse transformation.
        """
        pass


class PerSegmentWrapper(Transform):
    """Class to apply transform in per segment manner."""

    def __init__(self, transform: OneSegmentTransform, required_features: Union[Literal["all"], List[str]]):
        self._base_transform = transform
        self.segment_transforms: Optional[Dict[str, OneSegmentTransform]] = None
        super().__init__(required_features=required_features)

    def _fit(self, df: pd.DataFrame):
        """Fit transform on each segment."""
        self.segment_transforms = {}
        segments = df.columns.get_level_values("segment").unique()
        for segment in segments:
            self.segment_transforms[segment] = deepcopy(self._base_transform)
            self.segment_transforms[segment].fit(df[segment])

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to each segment separately."""
        if self.segment_transforms is None:
            raise ValueError("Transform is not fitted!")

        results = []
        segments = set(df.columns.get_level_values("segment"))
        for segment in segments:
            if segment not in self.segment_transforms:
                raise NotImplementedError("Per-segment transforms can't work on new segments!")

            segment_transform = self.segment_transforms[segment]
            seg_df = segment_transform.transform(df[segment])

            _idx = seg_df.columns.to_frame()
            _idx.insert(0, "segment", segment)
            seg_df.columns = pd.MultiIndex.from_frame(_idx)
            results.append(seg_df)

        df = pd.concat(results, axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return df


class IrreversiblePerSegmentWrapper(PerSegmentWrapper, IrreversibleTransform):
    """Class to apply irreversible transform in per segment manner."""

    def __init__(self, transform: OneSegmentTransform, required_features: Union[Literal["all"], List[str]]):
        super().__init__(transform=transform, required_features=required_features)


class ReversiblePerSegmentWrapper(PerSegmentWrapper, ReversibleTransform):
    """Class to apply reversible transform in per segment manner."""

    def __init__(self, transform: OneSegmentTransform, required_features: Union[Literal["all"], List[str]]):
        super().__init__(transform=transform, required_features=required_features)

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse_transform to each segment."""
        if self.segment_transforms is None:
            raise ValueError("Transform is not fitted!")

        results = []
        segments = set(df.columns.get_level_values("segment"))
        for segment in segments:
            if segment not in self.segment_transforms:
                raise NotImplementedError("Per-segment transforms can't work on new segments!")

            segment_transform = self.segment_transforms[segment]
            seg_df = segment_transform.inverse_transform(df[segment])

            _idx = seg_df.columns.to_frame()
            _idx.insert(0, "segment", segment)
            seg_df.columns = pd.MultiIndex.from_frame(_idx)

            results.append(seg_df)

        df = pd.concat(results, axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return df
