from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from ruptures.base import BaseEstimator

from etna.analysis.change_points_trend.search import _find_change_points_segment
from etna.transforms.base import FutureMixin
from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform

TTimestampInterval = Tuple[pd.Timestamp, pd.Timestamp]


class _OneSegmentChangePointSegmentationTransform(Transform):
    """_OneSegmentChangePointSegmentationTransform make label encoder to change points."""

    def __init__(
        self, in_column: str, change_point_model: BaseEstimator, out_column: str, **change_point_model_predict_params
    ):
        """Init _OneSegmentChangePointSegmentationTransform.
        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_point_model:
            model to get change points
        out_column: str, optional
            result column name. If not given use ``self.__repr__()``
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.out_column = out_column
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

    def _fill_per_interval(self, series: pd.Series) -> pd.Series:
        """Fill values in resulting series."""
        if self.intervals is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
        result_series = pd.Series(index=series.index)
        for k, interval in enumerate(self.intervals):
            tmp_series = series[interval[0] : interval[1]]
            if tmp_series.empty:
                continue
            result_series[tmp_series.index] = k
        return result_series.astype(int).astype("category")

    def fit(self, df: pd.DataFrame) -> "_OneSegmentChangePointSegmentationTransform":
        """Fit OneSegmentChangePointSegmentationTransform: find change points in ``df``.
        Parameters
        ----------
        df:
            one segment dataframe indexed with timestamp
        Returns
        -------
        :
        """
        series = df.loc[df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index(), self.in_column]
        if series.isnull().values.any():
            raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")
        change_points = _find_change_points_segment(
            series=series, change_point_model=self.change_point_model, **self.change_point_model_predict_params
        )
        self.intervals = self._build_intervals(change_points=change_points)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split df to intervals.
        Parameters
        ----------
        df:
            one segment dataframe
        Returns
        -------
        df: pd.DataFrame
            df with new column
        """
        series = df[self.in_column]
        result_series = self._fill_per_interval(series=series)
        df.loc[:, self.out_column] = result_series
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Do nothing in this case.
        Parameters
        ----------
        df:
            one segment dataframe
        Returns
        -------
        df: pd.DataFrame
            one segment dataframe
        """
        return df


class ChangePointSegmentationTransform(PerSegmentWrapper, FutureMixin):
    """ChangePointSegmentationTransform make label encoder to change points.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_point_model: BaseEstimator,
        out_column: Optional[str] = None,
        **change_point_model_predict_params,
    ):
        """Init ChangePointSegmentationTransform.

        Parameters
        ----------
        in_column:
            name of column to fit change point model
        change_point_model:
            model to get change points
        out_column: str, optional
            result column name. If not given use ``self.__repr__()``
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.out_column = out_column
        self.change_point_model = change_point_model
        self.change_point_model_predict_params = change_point_model_predict_params
        super().__init__(
            transform=_OneSegmentChangePointSegmentationTransform(
                in_column=self.in_column,
                out_column=self.out_column if self.out_column is not None else self.__repr__(),
                change_point_model=self.change_point_model,
                **self.change_point_model_predict_params,
            )
        )
