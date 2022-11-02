from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple
from typing import Type

import pandas as pd
from sklearn.base import RegressorMixin

from etna.core import BaseMixin

TTimestampInterval = Tuple[pd.Timestamp, pd.Timestamp]
TDetrendModel = Type[RegressorMixin]


class BaseChangePointsModelAdapter(BaseMixin, ABC):
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
