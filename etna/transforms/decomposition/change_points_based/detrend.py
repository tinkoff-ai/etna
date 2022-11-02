from typing import Optional

import numpy as np
import pandas as pd
from ruptures.detection import Binseg
from sklearn.linear_model import LinearRegression

from etna.transforms.decomposition.change_points_based.base import ReversibleChangePointsTransform
from etna.transforms.decomposition.change_points_based.base import _OneSegmentChangePointsTransform
from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.change_points_models import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_based.per_interval_models import PerIntervalModel
from etna.transforms.decomposition.change_points_based.per_interval_models import SklearnPerIntervalModel
from etna.transforms.utils import match_target_quantiles


class _OneSegmentChangePointsTrendTransform(_OneSegmentChangePointsTransform):
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


class ChangePointsTrendTransform(ReversibleChangePointsTransform):
    """ChangePointsTrendTransform uses :py:class:`ruptures.detection.Binseg` model as a change point detection model.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_points_model: Optional[BaseChangePointsModelAdapter] = None,
        per_interval_model: Optional[PerIntervalModel] = None,
    ):
        """Init ChangePointsTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_points_model:
            model to get trend change points
        per_interval_model:
            model to process intervals of segment
        """
        self.in_column = in_column
        self.change_points_model = (
            change_points_model
            if change_points_model is not None
            else RupturesChangePointsModel(
                change_points_model=Binseg(model="ar"),
                n_bkps=5,
            )
        )
        self.per_interval_model = (
            per_interval_model if per_interval_model is not None else SklearnPerIntervalModel(model=LinearRegression())
        )
        super().__init__(
            transform=_OneSegmentChangePointsTrendTransform(
                in_column=self.in_column,
                change_points_model=self.change_points_model,
                per_interval_model=self.per_interval_model,
            ),
            required_features=[in_column],
        )
