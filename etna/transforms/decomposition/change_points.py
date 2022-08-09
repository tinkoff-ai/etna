from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import pandas as pd
from ruptures.base import BaseEstimator
from sklearn.base import RegressorMixin

from etna.analysis.change_points_trend.search import _find_change_points_segment
from etna.transforms.base import Transform

TTimestampInterval = Tuple[pd.Timestamp, pd.Timestamp]
TDetrendModel = Type[RegressorMixin]


class _ChangePointsTransform(Transform):
    """_ChangePointsTransform is the base class for transforms with change points."""

    def __init__(
        self, in_column: str, out_column: str, change_point_model: BaseEstimator, **change_point_model_predict_params
    ):
        """Init _ChangePointsTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            result column name
        change_point_model:
            model to get change points
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.out_columns = out_column
        self.change_point_model = change_point_model
        self.intervals: Optional[List[TTimestampInterval]] = None
        self.change_point_model_predict_params = change_point_model_predict_params

    @staticmethod
    def _build_intervals(change_points: List[pd.Timestamp]) -> List[TTimestampInterval]:
        """Create list of stable intervals from list of change points."""
        change_points = sorted(change_points)
        left_border = pd.Timestamp.min
        intervals = []
        for point in change_points:
            right_border = point
            intervals.append((left_border, right_border))
            left_border = right_border
        intervals.append((left_border, pd.Timestamp.max))
        return intervals

    def fit(self, df: pd.DataFrame) -> "_ChangePointsTransform":
        """Fit _ChangePointsTransform: find change points in ``df`` and build intervals.

        Parameters
        ----------
        df:
            one segment dataframe indexed with timestamp

        Returns
        -------
        :

        Raises:
        -------
        ValueError
            If series contains NaNs in the middle
        """
        self.series = df.loc[
            df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index(), self.in_column
        ]
        if self.series.isnull().values.any():
            raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")
        change_points = _find_change_points_segment(
            series=self.series, change_point_model=self.change_point_model, **self.change_point_model_predict_params
        )
        self.intervals = self._build_intervals(change_points=change_points)
        return self
