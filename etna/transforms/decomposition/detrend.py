from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from etna.distributions import BaseDistribution
from etna.distributions import IntDistribution
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper
from etna.transforms.utils import match_target_quantiles


class _OneSegmentLinearTrendBaseTransform(OneSegmentTransform):
    """Transform for one segment that implements trend subtraction and reconstruction feature."""

    def __init__(self, in_column: str, regressor: RegressorMixin, poly_degree: int = 1):
        """
        Create instance of _OneSegmentLinearTrendBaseTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        regressor:
            instance of sklearn :py:class`sklearn.base.RegressorMixin` to predict trend
        poly_degree:
            degree of polynomial to fit trend on
        """
        self.in_column = in_column
        self.poly_degree = poly_degree
        self._pipeline = Pipeline(
            [("polynomial", PolynomialFeatures(degree=self.poly_degree, include_bias=False)), ("regressor", regressor)]
        )
        # verification that this variable is fitted isn't needed because this class isn't used by the user
        self._x_median = None

    @staticmethod
    def _get_x(df) -> np.ndarray:
        series_len = len(df)
        x = df.index.to_series()
        if isinstance(type(x.dtype), pd.Timestamp):
            raise ValueError("Your timestamp column has wrong format. Need np.datetime64 or datetime.datetime")
        x = x.apply(lambda ts: ts.timestamp())
        x = x.to_numpy().reshape(series_len, 1)
        return x

    def fit(self, df: pd.DataFrame) -> "_OneSegmentLinearTrendBaseTransform":
        """
        Fit regression detrend_model with data from df.

        Parameters
        ----------
        df:
            data that regressor should be trained with

        Returns
        -------
        _OneSegmentLinearTrendBaseTransform
            instance with trained regressor
        """
        df = df.dropna(subset=[self.in_column])
        x = self._get_x(df)
        self._x_median = np.median(x)
        x -= self._x_median
        y = df[self.in_column].tolist()
        self._pipeline.fit(x, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data from df: subtract linear trend found by regressor.

        Parameters
        ----------
        df:
            data to subtract trend from

        Returns
        -------
        pd.DataFrame
            residue after trend subtraction
        """
        result = df
        x = self._get_x(df)
        x -= self._x_median
        y = df[self.in_column].values
        trend = self._pipeline.predict(x)
        no_trend_timeseries = y - trend
        result[self.in_column] = no_trend_timeseries
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit regression detrend_model with data from df and subtract the trend from df.

        Parameters
        ----------
        df:
            data to train regressor and transform

        Returns
        -------
        pd.DataFrame
            residue after trend subtraction
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transformation for trend subtraction: add trend to prediction.

        Parameters
        ----------
        df:
            data to transform

        Returns
        -------
        pd.DataFrame
            data with reconstructed trend
        """
        result = df
        x = self._get_x(df)
        x -= self._x_median
        y = df[self.in_column].values
        trend = self._pipeline.predict(x)
        add_trend_timeseries = y + trend
        result[self.in_column] = add_trend_timeseries
        if self.in_column == "target":
            quantiles = match_target_quantiles(set(result.columns))
            for quantile_column_nm in quantiles:
                result.loc[:, quantile_column_nm] += trend
        return result


class LinearTrendTransform(ReversiblePerSegmentWrapper):
    """Transform that uses linear regression with polynomial features to make a detrending.

    Transform fits a :py:class:`sklearn.linear_model.LinearRegression` with polynomial features on each segment.
    Values predicted by the model are subtracted from each segment.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(self, in_column: str, poly_degree: int = 1, **regression_params):
        """Create instance of LinearTrendTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        poly_degree:
            degree of polynomial to fit trend on
        regression_params:
            params that should be used to init :py:class:`sklearn.linear_model.LinearRegression`
        """
        self.in_column = in_column
        self.poly_degree = poly_degree
        self.regression_params = regression_params
        super().__init__(
            transform=_OneSegmentLinearTrendBaseTransform(
                in_column=self.in_column,
                regressor=LinearRegression(**self.regression_params),
                poly_degree=self.poly_degree,
            ),
            required_features=[self.in_column],
        )

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        This grid tunes only ``poly_degree`` parameter. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "poly_degree": IntDistribution(low=1, high=2),
        }


class TheilSenTrendTransform(ReversiblePerSegmentWrapper):
    """Transform that uses Theil–Sen regression with polynomial features to make a detrending.

    Transform fits a :py:class:`sklearn.linear_model.TheilSenRegressor` with polynomial features on each segment.
    Values predicted by the model are subtracted from each segment.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.

    Notes
    -----
    Setting parameter ``n_subsamples`` manually might cause the error. It should be at least the number
    of features (plus 1 if ``fit_intercept=True``) and the number of samples in the shortest segment as a maximum.
    """

    def __init__(self, in_column: str, poly_degree: int = 1, **regression_params):
        """Create instance of TheilSenTrendTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        poly_degree:
            degree of polynomial to fit trend on
        regression_params:
            params that should be used to init :py:class:`sklearn.linear_model.TheilSenRegressor`
        """
        self.in_column = in_column
        self.poly_degree = poly_degree
        self.regression_params = regression_params
        super().__init__(
            transform=_OneSegmentLinearTrendBaseTransform(
                in_column=self.in_column,
                regressor=TheilSenRegressor(**self.regression_params),
                poly_degree=self.poly_degree,
            ),
            required_features=[self.in_column],
        )

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        This grid tunes ``poly_degree`` parameter. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "poly_degree": IntDistribution(low=1, high=2),
        }
