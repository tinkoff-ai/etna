from typing import List
from typing import Optional

import pandas as pd
from ruptures.base import BaseEstimator

from etna.transforms.base import FutureMixin
from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform
from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel
from etna.transforms.decomposition.base_change_points import TTimestampInterval


class _OneSegmentChangePointsSegmentationTransform(Transform):
    """_OneSegmentChangePointsSegmentationTransform make label encoder to change points."""

    def __init__(
        self, in_column: str, change_point_model: BaseEstimator, out_column: str, **change_point_model_predict_params
    ):
        """Init _OneSegmentChangePointsSegmentationTransform.
        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_point_model:
            model to get change points
        out_column:
            result column name. If not given use ``self.__repr__()``
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.out_column = out_column
        self.intervals: Optional[List[TTimestampInterval]] = None
        self.ruptures = RupturesChangePointsModel(change_point_model, **change_point_model_predict_params)

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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split df to intervals.

        Parameters
        ----------
        df:
            one segment dataframe

        Returns
        -------
        df:
            df with new column
        """
        series = df[self.in_column]
        result_series = self._fill_per_interval(series=series)
        df.loc[:, self.out_column] = result_series
        return df

    def fit(self, df: pd.DataFrame) -> "_OneSegmentChangePointsSegmentationTransform":
        """Fit _OneSegmentChangePointsSegmentationTransform: find change points in ``df`` and build intervals.

        Parameters
        ----------
        df:
            one segment dataframe indexed with timestamp

        Returns
        -------
        :

        Raises
        ------
        ValueError
            If series contains NaNs in the middle
        """
        self.intervals = self.ruptures.get_change_points(df=df, in_column=self.in_column)
        return self

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Do nothing in this case.

        Parameters
        ----------
        df:
            one segment dataframe

        Returns
        -------
        df:
            one segment dataframe
        """
        return df


class ChangePointsSegmentationTransform(PerSegmentWrapper, FutureMixin):
    """ChangePointsSegmentationTransform make label encoder to change points.

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
        """Init ChangePointsSegmentationTransform.

        Parameters
        ----------
        in_column:
            name of column to fit change point model
        change_point_model:
            model to get change points
        out_column:
            result column name. If not given use ``self.__repr__()``
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.in_column = in_column
        self.out_column = out_column
        self.change_point_model = change_point_model
        self.change_point_model_predict_params = change_point_model_predict_params
        if self.out_column is None:
            self.out_column = repr(self)
        super().__init__(
            transform=_OneSegmentChangePointsSegmentationTransform(
                in_column=self.in_column,
                out_column=self.out_column if self.out_column is not None else self.__repr__(),
                change_point_model=self.change_point_model,
                **self.change_point_model_predict_params,
            )
        )
