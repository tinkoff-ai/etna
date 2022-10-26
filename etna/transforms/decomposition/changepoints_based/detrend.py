from typing import List

import numpy as np
import pandas as pd
from ruptures.detection import Binseg
from sklearn.linear_model import LinearRegression

from etna.transforms import ReversiblePerSegmentWrapper
from etna.transforms.decomposition.changepoints_based.base import ChangePointsTransform
from etna.transforms.decomposition.changepoints_based.base import OneSegmentChangePointsTransform
from etna.transforms.decomposition.changepoints_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.changepoints_based.change_points_models import RupturesChangePointsModel
from etna.transforms.decomposition.changepoints_based.per_interval_models import PerIntervalModel
from etna.transforms.decomposition.changepoints_based.per_interval_models import SklearnPerIntervalModel
from etna.transforms.utils import match_target_quantiles


class _OneSegmentChangePointsTrendTransform(OneSegmentChangePointsTransform):
    """_OneSegmentChangePointsTransform subtracts multiple linear trend from series."""

    @staticmethod
    def _get_features(series: pd.Series) -> np.ndarray:
        """Convert ETNA timestamp-index to a list of timestamps to fit regression models."""
        timestamps = series.index
        timestamps = np.array([[ts.timestamp()] for ts in timestamps])
        return timestamps

    def _apply_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        df.loc[:, self.in_column] -= transformed_series
        return df

    def _apply_inverse_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        df.loc[:, self.in_column] += transformed_series
        if self.in_column == "target":
            quantiles = match_target_quantiles(set(df.columns))
            for quantile_column_nm in quantiles:
                df.loc[:, quantile_column_nm] += transformed_series
        return df


class ChangePointsTrendTransform(ChangePointsTransform, ReversiblePerSegmentWrapper):
    """ChangePointsTrendTransform uses :py:class:`ruptures.detection.Binseg` model as a change point detection model.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_point_model: BaseChangePointsModelAdapter = RupturesChangePointsModel(
            change_point_model=Binseg(model="ar"),
            n_bkps=5,
        ),
        per_interval_model: PerIntervalModel = SklearnPerIntervalModel(model=LinearRegression()),
    ):
        """Init ChangePointsTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_point_model:
            model to get trend change points
        per_interval_model:
            model to process intervals of segment
        """
        self.in_column = in_column
        self.change_point_model = change_point_model
        self.per_interval_model = per_interval_model
        super().__init__(
            transform=_OneSegmentChangePointsTrendTransform(
                in_column=self.in_column,
                change_point_model=self.change_point_model,
                per_interval_model=self.per_interval_model,
            ),
            required_features=[in_column],
        )

    def get_regressors_info(self) -> List[str]:
        return []
