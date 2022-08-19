from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple
from typing import Type

import pandas as pd
from ruptures.base import BaseEstimator
from ruptures.costs import CostLinear
from sklearn.base import RegressorMixin

TTimestampInterval = Tuple[pd.Timestamp, pd.Timestamp]
TDetrendModel = Type[RegressorMixin]


class BaseChangePointsModelAdapter(ABC):
    """BaseChangePointsModelAdapter is the base class for change point models adapters."""

    @abstractmethod
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
        pass

    @staticmethod
    def _build_intervals(change_points: List[pd.Timestamp]) -> List[TTimestampInterval]:
        """Create list of stable intervals from list of change points."""
        change_points.extend([pd.Timestamp.min, pd.Timestamp.max])
        change_points = sorted(change_points)
        intervals = list(zip(change_points[:-1], change_points[1:]))
        return intervals

    def get_change_points_intervals(self, df: pd.DataFrame, in_column: str) -> List[TTimestampInterval]:
        """Find change point intervals in given dataframe and column.

        Parameters
        ----------
        df:
            dataframe indexed with timestamp
        in_column:
            name of column to get change points

        Returns
        -------
        :
            change points intervals
        """
        change_points = self.get_change_points(df=df, in_column=in_column)
        intervals = self._build_intervals(change_points=change_points)
        return intervals


class RupturesChangePointsModel(BaseChangePointsModelAdapter):
    """RupturesChangePointsModel is ruptures change point models adapter."""

    def __init__(self, change_point_model: BaseEstimator, **change_point_model_predict_params):
        """Init RupturesChangePointsModel.

        Parameters
        ----------
        change_point_model:
            model to get change points
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.change_point_model = change_point_model
        self.model_predict_params = change_point_model_predict_params

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
        if isinstance(self.change_point_model.cost, CostLinear):
            signal = signal.reshape((-1, 1))
        timestamp = series.index
        self.change_point_model.fit(signal=signal)
        # last point in change points is the first index after the series
        change_points_indices = self.change_point_model.predict(**self.model_predict_params)[:-1]
        change_points = [timestamp[idx] for idx in change_points_indices]
        return change_points
