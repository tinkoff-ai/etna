from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import IrreversiblePerSegmentWrapper
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper
from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.per_interval_models import PerIntervalModel


class _OneSegmentChangePointsTransform(OneSegmentTransform, ABC):
    def __init__(
        self, in_column: str, change_points_model: BaseChangePointsModelAdapter, per_interval_model: PerIntervalModel
    ):
        """Init _OneSegmentChangePointsTransform.

        Parameters
        ----------
        in_column:
            name of column to apple transform to
        change_points_model:
            model to get change points from data
        per_interval_model:
            model to process intervals between change points
        """
        self.in_column = in_column
        self.change_points_model = change_points_model
        self.per_interval_model = per_interval_model
        self.per_interval_models: Optional[Dict[Any, PerIntervalModel]] = None
        self.intervals: Optional[List[Tuple[Any, Any]]] = None

    def _init_per_interval_models(self, intervals: List[Tuple[Any, Any]]) -> Dict[Tuple[Any, Any], PerIntervalModel]:
        """Multiply per interval model for given intervals."""
        per_interval_models = {interval: deepcopy(self.per_interval_model) for interval in intervals}
        return per_interval_models

    @staticmethod
    def _get_features(series: pd.Series) -> np.ndarray:
        """Prepare features to train per interval model.

        Parameters
        ----------
        series:
            series to get features from

        Returns
        -------
        features:
            array with prepared features
        """
        features = series.index.values.reshape((-1, 1))
        return features

    @staticmethod
    def _get_targets(series: pd.Series) -> np.ndarray:
        """Get targets from given series to train per interval model.

        Parameters
        ----------
        series:
            series to get targets from

        Returns
        -------
        targets:
            array with targets
        """
        return series.values

    def _fit_per_interval_models(self, series: pd.Series):
        """Fit per-interval models with corresponding data from series."""
        if self.intervals is None or self.per_interval_models is None:
            raise ValueError("Something went wrong on fit! Check the parameters of the transform.")
        for interval in self.intervals:
            tmp_series = series[interval[0] : interval[1]]
            features = self._get_features(series=tmp_series)
            targets = self._get_targets(series=tmp_series)
            self.per_interval_models[interval].fit(features=features, target=targets)

    def fit(self, df: pd.DataFrame) -> "_OneSegmentChangePointsTransform":
        """Fit transform.
        Get no-changepoints intervals with change_points_model and fit per_interval_model on the intervals.

        Parameters
        ----------
        df:
            dataframe to process

        Returns
        -------
        self:
            fitted _OneSegmentChangePointsTransform
        """
        self.intervals = self.change_points_model.get_change_points_intervals(df=df, in_column=self.in_column)
        self.per_interval_models = self._init_per_interval_models(intervals=self.intervals)

        series = df.loc[df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index(), self.in_column]
        self._fit_per_interval_models(series=series)
        return self

    def _predict_per_interval_model(self, series: pd.Series) -> pd.Series:
        """Apply per-interval detrending to series."""
        if self.intervals is None or self.per_interval_models is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
        prediction_series = pd.Series(index=series.index)
        for interval in self.intervals:
            tmp_series = series[interval[0] : interval[1]]
            if tmp_series.empty:
                continue
            features = self._get_features(series=tmp_series)
            per_interval_prediction = self.per_interval_models[interval].predict(features=features)
            prediction_series[tmp_series.index] = per_interval_prediction
        return prediction_series

    @abstractmethod
    def _apply_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        """Apply transformation given by per_interval_model.

        Parameters
        ----------
        df:
            original dataframe to apply transformation
        transformed_series:
            transformed series to be applied to df

        Returns
        -------
        transformed_df:
            dataframe with applied transformation
        """
        pass

    @abstractmethod
    def _apply_inverse_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        """Apply inverse transformation given by per_interval_model.

        Parameters
        ----------
        df:
            transformed dataframe to apply inverse transformation
        transformed_series:
            transformed series to be applied to df

        Returns
        -------
        transformed_df:
            dataframe with inverse transformation
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data from df.

        Parameters
        ----------
        df:
            dataframe to apply transformation to

        Returns
        -------
        transformed_df:
            dataframe with applied transformation
        """
        df._is_copy = False
        series = df[self.in_column]
        transformed_series = self._predict_per_interval_model(series=series)
        transformed_df = self._apply_transformation(df=df, transformed_series=transformed_series)
        return transformed_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split df to intervals of stable trend according to previous change point detection and add trend to each one.

        Parameters
        ----------
        df:
            one segment dataframe to turn trend back

        Returns
        -------
        df: pd.DataFrame
            df with restored trend in in_column
        """
        df._is_copy = False
        series = df[self.in_column]
        trend_series = self._predict_per_interval_model(series=series)
        self._apply_inverse_transformation(df=df, transformed_series=trend_series)
        return df


class BaseChangePointsTransform:
    """Base class for all the change points based transforms."""

    pass


class ReversibleChangePointsTransform(BaseChangePointsTransform, ReversiblePerSegmentWrapper):
    """ReversibleChangePointsTransform class is a base class for all reversible transforms that work with change point."""

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []


class IrreversibleChangePointsTransform(BaseChangePointsTransform, IrreversiblePerSegmentWrapper, FutureMixin):
    """IrreversibleChangePointsTransform class is a base class for all irreversible transforms that work with change point."""

    out_column: Optional[str] = None

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self.out_column]  # type: ignore
