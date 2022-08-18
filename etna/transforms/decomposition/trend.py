from typing import Optional

import pandas as pd
from ruptures import Binseg
from ruptures.base import BaseCost
from sklearn.linear_model import LinearRegression

from etna.transforms.base import FutureMixin
from etna.transforms.base import PerSegmentWrapper
from etna.transforms.decomposition.change_points_trend import BaseEstimator
from etna.transforms.decomposition.change_points_trend import TDetrendModel
from etna.transforms.decomposition.change_points_trend import _OneSegmentChangePointsTrendTransform


class _OneSegmentTrendTransform(_OneSegmentChangePointsTrendTransform):
    """_OneSegmentTrendTransform adds trend as a feature."""

    def __init__(
        self,
        in_column: str,
        out_column: str,
        change_point_model: BaseEstimator,
        detrend_model: TDetrendModel,
        **change_point_model_predict_params,
    ):
        """Init _OneSegmentTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column
        change_point_model:
            model to get trend change points
        detrend_model:
            model to get trend from data
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        self.out_column = out_column
        super().__init__(
            in_column=in_column,
            change_point_model=change_point_model,
            detrend_model=detrend_model,
            **change_point_model_predict_params,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add column with trend, got from the detrend_model.

        Parameters
        ----------
        df:
            data to get trend from

        Returns
        -------
        pd.DataFrame:
            df with trend column
        """
        df._is_copy = False
        series = df[self.in_column]
        trend_series = self._predict_per_interval_model(series=series)
        df[self.out_column] = trend_series
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform dataframe.

        Parameters
        ----------
        df:
            one segment dataframe

        Returns
        -------
        pd.DataFrame:
            given dataframe
        """
        return df


class _TrendTransform(PerSegmentWrapper):
    """_TrendTransform adds trend as a feature. Creates column '<in_column>_trend'."""

    def __init__(
        self,
        in_column: str,
        out_column: str,
        change_point_model: BaseEstimator,
        detrend_model: TDetrendModel,
        **change_point_model_predict_params,
    ):
        """Init _TrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column
        change_point_model:
            model to get trend change points
        detrend_model:
            model to get trend in data
        change_point_model_predict_params:
            params for ``change_point_model.predict`` method
        """
        super().__init__(
            transform=_OneSegmentTrendTransform(
                in_column=in_column,
                out_column=out_column,
                change_point_model=change_point_model,
                detrend_model=detrend_model,
                **change_point_model_predict_params,
            )
        )


class TrendTransform(_TrendTransform, FutureMixin):
    """TrendTransform adds trend as a feature.

    TrendTransform uses uses :py:class:`ruptures.detection.Binseg` model as a change point detection model
    in _TrendTransform.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        out_column: Optional[str] = None,
        detrend_model: Optional[TDetrendModel] = None,
        model: str = "ar",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 1,
        n_bkps: int = 5,
        pen: Optional[float] = None,
        epsilon: Optional[float] = None,
    ):
        """Init TrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column.
            If not given, use ``self.__repr__()``
        detrend_model:
            model to get trend in data
        model:
            binseg segment model, ["l1", "l2", "rbf",...]. Not used if 'custom_cost' is not None.
        custom_cost:
            binseg custom cost function
        min_size:
            minimum segment length necessary to decide it is a stable trend segment
        jump:
            jump value can speed up computations: if ``jump==k``,
            the algo will use every k-th value for change points search.
        n_bkps:
            number of change points to find
        pen:
            penalty value (>0)
        epsilon:
            reconstruction budget (>0)
        """
        self.in_column = in_column
        self.out_column = out_column
        self.detrend_model = LinearRegression() if detrend_model is None else detrend_model
        self.model = model
        self.custom_cost = custom_cost
        self.min_size = min_size
        self.jump = jump
        self.n_bkps = n_bkps
        self.pen = pen
        self.epsilon = epsilon
        super().__init__(
            in_column=self.in_column,
            out_column=self.out_column if self.out_column is not None else f"{self.__repr__()}",
            change_point_model=Binseg(
                model=self.model, custom_cost=self.custom_cost, min_size=self.min_size, jump=self.jump
            ),
            detrend_model=self.detrend_model,
            n_bkps=self.n_bkps,
            pen=self.pen,
            epsilon=self.epsilon,
        )
