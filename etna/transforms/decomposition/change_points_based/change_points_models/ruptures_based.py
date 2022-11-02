from typing import List

import pandas as pd
from ruptures.base import BaseEstimator
from ruptures.costs import CostLinear

from etna.transforms.decomposition.change_points_based.change_points_models.base import BaseChangePointsModelAdapter


class RupturesChangePointsModel(BaseChangePointsModelAdapter):
    """RupturesChangePointsModel is ruptures change point models adapter."""

    def __init__(self, change_points_model: BaseEstimator, **change_points_model_predict_params):
        """Init RupturesChangePointsModel.
        @ TODO: add arg names for pen/eps/n_bkps or clear error message about its validation

        Parameters
        ----------
        change_point_model:
            model to get change points
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.change_points_model = change_points_model
        self.change_points_model_predict_params = change_points_model_predict_params

    def get_change_points(self, df: pd.DataFrame, in_column: str) -> List[pd.Timestamp]:
        """Find change points within one segment.

        Parameters
        ----------
        df:
            dataframe indexed with timestamp
        in_column:
            name of column to get change points

        Returns
        -------
        change points:
            change point timestamps
        """
        series = df.loc[df[in_column].first_valid_index() : df[in_column].last_valid_index(), in_column]
        if series.isnull().values.any():
            raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")

        signal = series.to_numpy()
        if isinstance(self.change_points_model.cost, CostLinear):
            signal = signal.reshape((-1, 1))
        timestamp = series.index
        self.change_points_model.fit(signal=signal)
        # last point in change points is the first index after the series
        change_points_indices = self.change_points_model.predict(**self.change_points_model_predict_params)[:-1]
        change_points = [timestamp[idx] for idx in change_points_indices]
        return change_points
