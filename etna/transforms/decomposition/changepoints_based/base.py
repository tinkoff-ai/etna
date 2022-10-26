from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import ReversiblePerSegmentWrapper
from etna.transforms.decomposition.changepoints_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.changepoints_based.per_interval_models import PerIntervalModel


class OneSegmentChangePointsTransform(OneSegmentTransform):
    def __init__(
        self, in_column: str, change_point_model: BaseChangePointsModelAdapter, per_interval_model: PerIntervalModel
    ):
        self.in_column = in_column
        self.change_point_model = change_point_model
        self.per_interval_model = per_interval_model
        self.per_interval_models: Optional[Dict[Any, PerIntervalModel]] = None
        self.intervals: Optional[List[Tuple[Any, Any]]] = None

    def _init_per_interval_models(self, intervals: List[Tuple[Any, Any]]):
        per_interval_models = {interval: deepcopy(self.per_interval_model) for interval in intervals}
        return per_interval_models

    @staticmethod
    def _get_features(series: pd.Series) -> np.ndarray:
        features = series.index.values.reshape((-1, 1))
        return features

    @staticmethod
    def _get_targets(series: pd.Series) -> np.ndarray:
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

    def fit(self, df: pd.DataFrame) -> "OneSegmentChangePointsTransform":

        self.intervals = self.change_point_model.get_change_points_intervals(df=df, in_column=self.in_column)
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

    def _apply_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        return df

    def _apply_inverse_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df._is_copy = False
        series = df[self.in_column]
        transformed_series = self._predict_per_interval_model(series=series)
        df = self._apply_transformation(df=df, transformed_series=transformed_series)
        return df

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


class ChangePointsTransform(PerSegmentWrapper):
    """ChangePointsTrendTransform subtracts multiple linear trend from series.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []
