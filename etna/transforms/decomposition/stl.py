from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.forecasting.stl import STLForecastResults

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform
from etna.transforms.utils import match_target_quantiles


class _OneSegmentSTLTransform(Transform):
    def __init__(
        self,
        in_column: str,
        period: int,
        model: Union[str, TimeSeriesModel] = "arima",
        robust: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        stl_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Init _OneSegmentSTLTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            model to predict trend, default options are:

            1. "arima": ``ARIMA(data, 1, 1, 0)`` (default)

            2. "holt": ``ETSModel(data, trend='add')``

            Custom model should be a subclass of :py:class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel`
            and have method ``get_prediction`` (not just ``predict``)
        robust:
            flag indicating whether to use robust version of STL
        model_kwargs:
            parameters for the model like in :py:class:`statsmodels.tsa.seasonal.STLForecast`
        stl_kwargs:
            additional parameters for :py:class:`statsmodels.tsa.seasonal.STLForecast`
        """
        if model_kwargs is None:
            model_kwargs = {}
        if stl_kwargs is None:
            stl_kwargs = {}

        self.in_column = in_column
        self.period = period

        if isinstance(model, str):
            if model == "arima":
                self.model = ARIMA
                if len(model_kwargs) == 0:
                    model_kwargs = {"order": (1, 1, 0)}
            elif model == "holt":
                self.model = ETSModel
                if len(model_kwargs) == 0:
                    model_kwargs = {"trend": "add"}
            else:
                raise ValueError(f"Not a valid option for model: {model}")
        elif isinstance(model, TimeSeriesModel):
            self.model = model
        else:
            raise ValueError("Model should be a string or TimeSeriesModel")

        self.robust = robust
        self.model_kwargs = model_kwargs
        self.stl_kwargs = stl_kwargs
        self.fit_results: Optional[STLForecastResults] = None

    def fit(self, df: pd.DataFrame) -> "_OneSegmentSTLTransform":
        """
        Perform STL decomposition and fit trend model.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: _OneSegmentSTLTransform
            instance after processing
        """
        df = df.loc[df[self.in_column].first_valid_index() : df[self.in_column].last_valid_index()]
        if df[self.in_column].isnull().values.any():
            raise ValueError("The input column contains NaNs in the middle of the series! Try to use the imputer.")
        model = STLForecast(
            df[self.in_column],
            self.model,
            model_kwargs=self.model_kwargs,
            period=self.period,
            robust=self.robust,
            **self.stl_kwargs,
        )
        self.fit_results = model.fit()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtract trend and seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: pd.DataFrame
            Dataframe with extracted features
        """
        result = df.copy()
        if self.fit_results is not None:
            season_trend = self.fit_results.get_prediction(
                start=df[self.in_column].first_valid_index(), end=df[self.in_column].last_valid_index()
            ).predicted_mean
        else:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
        result[self.in_column] -= season_trend
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend and seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: pd.DataFrame
            Dataframe with extracted features
        """
        result = df.copy()
        if self.fit_results is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling inverse_transform method.")
        season_trend = self.fit_results.get_prediction(
            start=df[self.in_column].first_valid_index(), end=df[self.in_column].last_valid_index()
        ).predicted_mean
        result[self.in_column] += season_trend
        if self.in_column == "target":
            quantiles = match_target_quantiles(set(result.columns))
            for quantile_column_nm in quantiles:
                result.loc[:, quantile_column_nm] += season_trend
        return result


class STLTransform(PerSegmentWrapper):
    """Transform that uses :py:class:`statsmodels.tsa.seasonal.STL` to subtract season and trend from the data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        period: int,
        model: Union[str, TimeSeriesModel] = "arima",
        robust: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        stl_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Init STLTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            model to predict trend, default options are:

            1. "arima": ``ARIMA(data, 1, 1, 0)`` (default)

            2. "holt": ``ETSModel(data, trend='add')``

            Custom model should be a subclass of :py:class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel`
            and have method ``get_prediction`` (not just ``predict``)
        robust:
            flag indicating whether to use robust version of STL
        model_kwargs:
            parameters for the model like in :py:class:`statsmodels.tsa.seasonal.STLForecast`
        stl_kwargs:
            additional parameters for :py:class:`statsmodels.tsa.seasonal.STLForecast`
        """
        self.in_column = in_column
        self.period = period
        self.model = model
        self.robust = robust
        self.model_kwargs = model_kwargs
        self.stl_kwargs = stl_kwargs
        super().__init__(
            transform=_OneSegmentSTLTransform(
                in_column=self.in_column,
                period=self.period,
                model=self.model,
                robust=self.robust,
                model_kwargs=self.model_kwargs,
                stl_kwargs=self.stl_kwargs,
            )
        )
