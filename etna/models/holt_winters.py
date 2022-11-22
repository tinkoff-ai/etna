import warnings
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper

from etna.models.base import BaseAdapter
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PerSegmentModelMixin


class _HoltWintersAdapter(BaseAdapter):
    """
    Class for holding Holt-Winters' exponential smoothing model.

    Notes
    -----
    We use :py:class:`statsmodels.tsa.holtwinters.ExponentialSmoothing` model from statsmodels package.
    """

    def __init__(
        self,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        initialization_method: str = "estimated",
        initial_level: Optional[float] = None,
        initial_trend: Optional[float] = None,
        initial_seasonal: Optional[Sequence[float]] = None,
        use_boxcox: Union[bool, str, float] = False,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        dates: Optional[Sequence[datetime]] = None,
        freq: Optional[str] = None,
        missing: str = "none",
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
        damping_trend: Optional[float] = None,
        **fit_kwargs,
    ):
        """
        Init Holt-Winters' model with given params.

        Parameters
        ----------
        trend:
            Type of trend component. One of:

            * 'add'

            * 'mul'

            * 'additive'

            * 'multiplicative'

            * None

        damped_trend:
            Should the trend component be damped.
        seasonal:
            Type of seasonal component. One of:

            * 'add'

            * 'mul'

            * 'additive'

            * 'multiplicative'

            * None

        seasonal_periods:
            The number of periods in a complete seasonal cycle, e.g., 4 for
            quarterly data or 7 for daily data with a weekly cycle.
        initialization_method:
            Method for initialize the recursions. One of:

            * None

            * 'estimated'

            * 'heuristic'

            * 'legacy-heuristic'

            * 'known'

            None defaults to the pre-0.12 behavior where initial values
            are passed as part of ``fit``. If any of the other values are
            passed, then the initial values must also be set when constructing
            the model. If 'known' initialization is used, then ``initial_level``
            must be passed, as well as ``initial_trend`` and ``initial_seasonal`` if
            applicable. Default is 'estimated'. "legacy-heuristic" uses the same
            values that were used in statsmodels 0.11 and earlier.
        initial_level:
            The initial level component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        initial_trend:
            The initial trend component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        initial_seasonal:
            The initial seasonal component. An array of length `seasonal`
            or length ``seasonal - 1`` (in which case the last initial value
            is computed to make the average effect zero). Only used if
            initialization is 'known'. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        use_boxcox: {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? One of:

            * True

            * False

            * 'log': apply log

            * float: lambda value

        bounds:
            An dictionary containing bounds for the parameters in the model,
            excluding the initial values if estimated. The keys of the dictionary
            are the variable names, e.g., smoothing_level or initial_slope.
            The initial seasonal variables are labeled initial_seasonal.<j>
            for j=0,...,m-1 where m is the number of period in a full season.
            Use None to indicate a non-binding constraint, e.g., (0, None)
            constrains a parameter to be non-negative.
        dates:
            An array-like object of datetime objects. If a Pandas object is given
            for endog, it is assumed to have a DateIndex.
        freq:
            The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
            'M', 'A', or 'Q'. This is optional if dates are given.
        missing:
            Available options are 'none', 'drop', and 'raise'. If 'none', no nan
            checking is done. If 'drop', any observations with nans are dropped.
            If 'raise', an error is raised. Default is 'none'.
        smoothing_level:
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend:
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        smoothing_seasonal:
            The gamma value of the holt winters seasonal method, if the value
            is set then this value will be used as the value.
        damping_trend:
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        fit_kwargs:
            Additional parameters for calling :py:meth:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit`.
        """
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.use_boxcox = use_boxcox
        self.bounds = bounds
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.damping_trend = damping_trend
        self.fit_kwargs = fit_kwargs

        self._model: Optional[ExponentialSmoothing] = None
        self._result: Optional[HoltWintersResultsWrapper] = None

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_HoltWintersAdapter":
        """
        Fit Holt-Winters' model.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors(ignored in this model)
        Returns
        -------
        :
            Fitted model
        """
        self._check_df(df)

        targets = df["target"]
        targets.index = df["timestamp"]

        self._model = ExponentialSmoothing(
            endog=targets,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method=self.initialization_method,
            initial_level=self.initial_level,
            initial_trend=self.initial_trend,
            initial_seasonal=self.initial_seasonal,
            use_boxcox=self.use_boxcox,
            bounds=self.bounds,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
        )
        self._result = self._model.fit(
            smoothing_level=self.smoothing_level,
            smoothing_trend=self.smoothing_trend,
            smoothing_seasonal=self.smoothing_seasonal,
            damping_trend=self.damping_trend,
            **self.fit_kwargs,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predictions from a Holt-Winters' model.

        Parameters
        ----------
        df:
            Features dataframe

        Returns
        -------
        :
            Array with predictions
        """
        if self._result is None or self._model is None:
            raise ValueError("This model is not fitted! Fit the model before calling predict method!")
        self._check_df(df)

        forecast = self._result.predict(start=df["timestamp"].min(), end=df["timestamp"].max())
        y_pred = forecast.values
        return y_pred

    def _check_df(self, df: pd.DataFrame):
        columns = df.columns
        columns_not_used = set(columns).difference({"target", "timestamp"})
        if columns_not_used:
            warnings.warn(
                message=f"This model does not work with exogenous features and regressors.\n "
                f"{columns_not_used} will be dropped"
            )

    def get_model(self) -> HoltWintersResultsWrapper:
        """Get :py:class:`statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper` model that was fitted inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._result


class HoltWintersModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """
    Holt-Winters' etna model.

    Notes
    -----
    We use :py:class:`statsmodels.tsa.holtwinters.ExponentialSmoothing` model from statsmodels package.
    """

    def __init__(
        self,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        initialization_method: str = "estimated",
        initial_level: Optional[float] = None,
        initial_trend: Optional[float] = None,
        initial_seasonal: Optional[Sequence[float]] = None,
        use_boxcox: Union[bool, str, float] = False,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        dates: Optional[Sequence[datetime]] = None,
        freq: Optional[str] = None,
        missing: str = "none",
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
        damping_trend: Optional[float] = None,
        **fit_kwargs,
    ):
        """
        Init Holt-Winters' model with given params.

        Parameters
        ----------
        trend:
            Type of trend component. One of:

            * 'add'

            * 'mul'

            * 'additive'

            * 'multiplicative'

            * None

        damped_trend:
            Should the trend component be damped.
        seasonal:
            Type of seasonal component. One of:

            * 'add'

            * 'mul'

            * 'additive'

            * 'multiplicative'

            * None

        seasonal_periods:
            The number of periods in a complete seasonal cycle, e.g., 4 for
            quarterly data or 7 for daily data with a weekly cycle.
        initialization_method:
            Method for initialize the recursions. One of:

            * None

            * 'estimated'

            * 'heuristic'

            * 'legacy-heuristic'

            * 'known'

            None defaults to the pre-0.12 behavior where initial values
            are passed as part of ``fit``. If any of the other values are
            passed, then the initial values must also be set when constructing
            the model. If 'known' initialization is used, then ``initial_level``
            must be passed, as well as ``initial_trend`` and ``initial_seasonal`` if
            applicable. Default is 'estimated'. "legacy-heuristic" uses the same
            values that were used in statsmodels 0.11 and earlier.
        initial_level:
            The initial level component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        initial_trend:
            The initial trend component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        initial_seasonal:
            The initial seasonal component. An array of length `seasonal`
            or length ``seasonal - 1`` (in which case the last initial value
            is computed to make the average effect zero). Only used if
            initialization is 'known'. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        use_boxcox: {True, False, 'log', float}, optional
            Should the Box-Cox transform be applied to the data first? One of:

            * True

            * False

            * 'log': apply log

            * float: lambda value

        bounds:
            An dictionary containing bounds for the parameters in the model,
            excluding the initial values if estimated. The keys of the dictionary
            are the variable names, e.g., smoothing_level or initial_slope.
            The initial seasonal variables are labeled initial_seasonal.<j>
            for j=0,...,m-1 where m is the number of period in a full season.
            Use None to indicate a non-binding constraint, e.g., (0, None)
            constrains a parameter to be non-negative.
        dates:
            An array-like object of datetime objects. If a Pandas object is given
            for endog, it is assumed to have a DateIndex.
        freq:
            The frequency of the time-series. A Pandas offset or 'B', 'D', 'W',
            'M', 'A', or 'Q'. This is optional if dates are given.
        missing:
            Available options are 'none', 'drop', and 'raise'. If 'none', no nan
            checking is done. If 'drop', any observations with nans are dropped.
            If 'raise', an error is raised. Default is 'none'.
        smoothing_level:
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend:
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        smoothing_seasonal:
            The gamma value of the holt winters seasonal method, if the value
            is set then this value will be used as the value.
        damping_trend:
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        fit_kwargs:
            Additional parameters for calling :py:meth:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit`.
        """
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.use_boxcox = use_boxcox
        self.bounds = bounds
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.damping_trend = damping_trend
        self.fit_kwargs = fit_kwargs
        super().__init__(
            base_model=_HoltWintersAdapter(
                trend=self.trend,
                damped_trend=self.damped_trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method=self.initialization_method,
                initial_level=self.initial_level,
                initial_trend=self.initial_trend,
                initial_seasonal=self.initial_seasonal,
                use_boxcox=self.use_boxcox,
                bounds=self.bounds,
                dates=self.dates,
                freq=self.freq,
                missing=self.missing,
                smoothing_level=self.smoothing_level,
                smoothing_trend=self.smoothing_trend,
                smoothing_seasonal=self.smoothing_seasonal,
                damping_trend=self.damping_trend,
                **self.fit_kwargs,
            )
        )


class HoltModel(HoltWintersModel):
    """
    Holt etna model.

    Restricted version of HoltWinters model.

    Notes
    -----
    We use :py:class:`statsmodels.tsa.holtwinters.ExponentialSmoothing` model from statsmodels package.
    They implement :py:class:`statsmodels.tsa.holtwinters.Holt` model
    as a restricted version of :py:class:`~statsmodels.tsa.holtwinters.ExponentialSmoothing` model.
    """

    def __init__(
        self,
        exponential: bool = False,
        damped_trend: bool = False,
        initialization_method: str = "estimated",
        initial_level: Optional[float] = None,
        initial_trend: Optional[float] = None,
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        damping_trend: Optional[float] = None,
        **fit_kwargs,
    ):
        """
        Init Holt model with given params.

        Parameters
        ----------
        exponential:
            Type of trend component. One of:

            * False: additive trend

            * True: multiplicative trend

        damped_trend:
            Should the trend component be damped.
        initialization_method:
            Method for initialize the recursions. One of:

            * None

            * 'estimated'

            * 'heuristic'

            * 'legacy-heuristic'

            * 'known'

            None defaults to the pre-0.12 behavior where initial values
            are passed as part of ``fit``. If any of the other values are
            passed, then the initial values must also be set when constructing
            the model. If 'known' initialization is used, then ``initial_level``
            must be passed, as well as ``initial_trend`` and ``initial_seasonal`` if
            applicable. Default is 'estimated'. "legacy-heuristic" uses the same
            values that were used in statsmodels 0.11 and earlier.
        initial_level:
            The initial level component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        initial_trend:
            The initial trend component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        smoothing_level:
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        smoothing_trend:
            The beta value of the Holt's trend method, if the value is
            set then this value will be used as the value.
        damping_trend:
            The phi value of the damped method, if the value is
            set then this value will be used as the value.
        fit_kwargs:
            Additional parameters for calling :py:meth:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit`.
        """
        trend = "mul" if exponential else "add"
        super().__init__(
            trend=trend,
            damped_trend=damped_trend,
            initialization_method=initialization_method,
            initial_level=initial_level,
            initial_trend=initial_trend,
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            damping_trend=damping_trend,
            **fit_kwargs,
        )


class SimpleExpSmoothingModel(HoltWintersModel):
    """
    Exponential smoothing etna model.

    Restricted version of HoltWinters model.

    Notes
    -----
    We use :py:class:`statsmodels.tsa.holtwinters.ExponentialSmoothing` model from statsmodels package.
    They implement :py:class:`statsmodels.tsa.holtwinters.SimpleExpSmoothing` model
    as a restricted version of :py:class:`~statsmodels.tsa.holtwinters.ExponentialSmoothing` model.
    """

    def __init__(
        self,
        initialization_method: str = "estimated",
        initial_level: Optional[float] = None,
        smoothing_level: Optional[float] = None,
        **fit_kwargs,
    ):
        """
        Init Exponential smoothing model with given params.

        Parameters
        ----------
        initialization_method:
            Method for initialize the recursions. One of:

            * None

            * 'estimated'

            * 'heuristic'

            * 'legacy-heuristic'

            * 'known'

            None defaults to the pre-0.12 behavior where initial values
            are passed as part of ``fit``. If any of the other values are
            passed, then the initial values must also be set when constructing
            the model. If 'known' initialization is used, then ``initial_level``
            must be passed, as well as ``initial_trend`` and ``initial_seasonal`` if
            applicable. Default is 'estimated'. "legacy-heuristic" uses the same
            values that were used in statsmodels 0.11 and earlier.
        initial_level:
            The initial level component. Required if estimation method is "known".
            If set using either "estimated" or "heuristic" this value is used.
            This allows one or more of the initial values to be set while
            deferring to the heuristic for others or estimating the unset
            parameters.
        smoothing_level:
            The alpha value of the simple exponential smoothing, if the value
            is set then this value will be used as the value.
        fit_kwargs:
            Additional parameters for calling :py:meth:`statsmodels.tsa.holtwinters.ExponentialSmoothing.fit`.
        """
        super().__init__(
            initialization_method=initialization_method,
            initial_level=initial_level,
            smoothing_level=smoothing_level,
            **fit_kwargs,
        )
