from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
import pandas as pd
from ruptures.base import BaseEstimator
from sklearn.base import RegressorMixin

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform
from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel
from etna.transforms.decomposition.base_change_points import TTimestampInterval
from etna.transforms.utils import match_target_quantiles

TDetrendModel = Type[RegressorMixin]


class _OneSegmentChangePointsTrendTransform(Transform):
    """_OneSegmentChangePointsTransform subtracts multiple linear trend from series."""

    def __init__(
        self,
        in_column: str,
        change_point_model: BaseEstimator,
        detrend_model: TDetrendModel,
        **change_point_model_predict_params,
    ):
        """Init _OneSegmentChangePointsTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_point_model:
            model to get trend change points
            TODO: replace this parameters with the instance of BaseChangePointsModelAdapter in ETNA 2.0
        detrend_model:
            model to get trend in data
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.out_columns = in_column
        self.ruptures_change_point_model = RupturesChangePointsModel(
            change_point_model=change_point_model, **change_point_model_predict_params
        )
        self.detrend_model = detrend_model
        self.per_interval_models: Optional[Dict[TTimestampInterval, TDetrendModel]] = None
        self.intervals: Optional[List[TTimestampInterval]] = None
        self.change_point_model = change_point_model
        self.change_point_model_predict_params = change_point_model_predict_params

    def _init_detrend_models(
        self, intervals: List[TTimestampInterval]
    ) -> Dict[Tuple[pd.Timestamp, pd.Timestamp], TDetrendModel]:
        """Create copy of detrend model for each timestamp interval."""
        per_interval_models = {interval: deepcopy(self.detrend_model) for interval in intervals}
        return per_interval_models

    def _get_timestamps(self, series: pd.Series) -> np.ndarray:
        """Convert ETNA timestamp-index to a list of timestamps to fit regression models."""
        timestamps = series.index
        timestamps = np.array([[ts.timestamp()] for ts in timestamps])
        return timestamps

    def _fit_per_interval_model(self, series: pd.Series):
        """Fit per-interval models with corresponding data from series."""
        if self.intervals is None or self.per_interval_models is None:
            raise ValueError("Something went wrong on fit! Check the parameters of the transform.")
        for interval in self.intervals:
            tmp_series = series[interval[0] : interval[1]]
            x = self._get_timestamps(series=tmp_series)
            y = tmp_series.values
            self.per_interval_models[interval].fit(x, y)

    def _predict_per_interval_model(self, series: pd.Series) -> pd.Series:
        """Apply per-interval detrending to series."""
        if self.intervals is None or self.per_interval_models is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
        trend_series = pd.Series(index=series.index)
        for interval in self.intervals:
            tmp_series = series[interval[0] : interval[1]]
            if tmp_series.empty:
                continue
            x = self._get_timestamps(series=tmp_series)
            trend = self.per_interval_models[interval].predict(x)
            trend_series[tmp_series.index] = trend
        return trend_series

    def fit(self, df: pd.DataFrame) -> "_OneSegmentChangePointsTrendTransform":
        """Fit OneSegmentChangePointsTransform: find trend change points in ``df``, fit detrend models with data from intervals of stable trend.

        Parameters
        ----------
        df:
            one segment dataframe indexed with timestamp

        Returns
        -------
        :
        """
        self.intervals = self.ruptures_change_point_model.get_change_points_intervals(df=df, in_column=self.in_column)
        self.per_interval_models = self._init_detrend_models(intervals=self.intervals)

        series = df.loc[df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index(), self.in_column]
        self._fit_per_interval_model(series=series)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split df to intervals of stable trend and subtract trend from each one.

        Parameters
        ----------
        df:
            one segment dataframe to subtract trend

        Returns
        -------
        detrended df: pd.DataFrame
            df with detrended in_column series
        """
        df._is_copy = False
        series = df[self.in_column]
        trend_series = self._predict_per_interval_model(series=series)
        df.loc[:, self.in_column] -= trend_series
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
        df.loc[:, self.in_column] += trend_series
        if self.in_column == "target":
            quantiles = match_target_quantiles(set(df.columns))
            for quantile_column_nm in quantiles:
                df.loc[:, quantile_column_nm] += trend_series
        return df


class ChangePointsTrendTransform(PerSegmentWrapper):
    """ChangePointsTrendTransform subtracts multiple linear trend from series.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_point_model: BaseEstimator,
        detrend_model: TDetrendModel,
        **change_point_model_predict_params,
    ):
        """Init ChangePointsTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_point_model:
            model to get trend change points
            TODO: replace this parameters with the instance of BaseChangePointsModelAdapter in ETNA 2.0
        detrend_model:
            model to get trend in data
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.change_point_model = change_point_model
        self.detrend_model = detrend_model
        self.change_point_model_predict_params = change_point_model_predict_params
        super().__init__(
            transform=_OneSegmentChangePointsTrendTransform(
                in_column=self.in_column,
                change_point_model=self.change_point_model,
                detrend_model=self.detrend_model,
                **self.change_point_model_predict_params,
            )
        )
