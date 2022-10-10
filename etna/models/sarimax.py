import warnings
from abc import abstractmethod
from datetime import datetime
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from etna.libs.pmdarima_utils import seasonal_prediction_with_confidence
from etna.models.base import BaseAdapter
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.utils import determine_num_steps

warnings.filterwarnings(
    message="No frequency information was provided, so inferred frequency .* will be used",
    action="ignore",
    category=ValueWarning,
    module="statsmodels.tsa.base.tsa_model",
)


class _SARIMAXBaseAdapter(BaseAdapter):
    """Base class for adapters based on :py:class:`statsmodels.tsa.statespace.sarimax.SARIMAX`."""

    def __init__(self):
        self.regressor_columns = None
        self._fit_results = None
        self._freq = None
        self._first_train_timestamp = None

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_SARIMAXBaseAdapter":
        """
        Fits a SARIMAX model.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors

        Returns
        -------
        :
            Fitted model
        """
        self.regressor_columns = regressors

        self._encode_categoricals(df)
        self._check_df(df)

        exog_train = self._select_regressors(df)
        self._fit_results = self._get_fit_results(endog=df["target"], exog=exog_train)

        freq = pd.infer_freq(df["timestamp"], warn=False)
        if freq is None:
            raise ValueError("Can't determine frequency of a given dataframe")
        self._freq = freq
        self._first_train_timestamp = df["timestamp"].min()

        return self

    def _make_prediction(
        self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float], dynamic: bool
    ) -> pd.DataFrame:
        """Make predictions taking into account ``dynamic`` parameter."""
        if self._fit_results is None:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        horizon = len(df)
        self._encode_categoricals(df)
        self._check_df(df, horizon)

        exog_future = self._select_regressors(df)
        start_timestamp = df["timestamp"].min()
        end_timestamp = df["timestamp"].max()
        # determine index of start_timestamp if counting from first timestamp of train
        start_idx = determine_num_steps(
            start_timestamp=self._first_train_timestamp, end_timestamp=start_timestamp, freq=self._freq  # type: ignore
        )
        # determine index of end_timestamp if counting from first timestamp of train
        end_idx = determine_num_steps(
            start_timestamp=self._first_train_timestamp, end_timestamp=end_timestamp, freq=self._freq  # type: ignore
        )

        if prediction_interval:
            forecast, _ = seasonal_prediction_with_confidence(
                arima_res=self._fit_results, start=start_idx, end=end_idx, X=exog_future, alpha=0.05, dynamic=dynamic
            )
            y_pred = pd.DataFrame({"mean": forecast})
            for quantile in quantiles:
                # set alpha in the way to get a desirable quantile
                alpha = min(quantile * 2, (1 - quantile) * 2)
                _, borders = seasonal_prediction_with_confidence(
                    arima_res=self._fit_results,
                    start=start_idx,
                    end=end_idx,
                    X=exog_future,
                    alpha=alpha,
                    dynamic=dynamic,
                )
                if quantile < 1 / 2:
                    series = borders[:, 0]
                else:
                    series = borders[:, 1]
                y_pred[f"mean_{quantile:.4g}"] = series
        else:
            forecast, _ = seasonal_prediction_with_confidence(
                arima_res=self._fit_results, start=start_idx, end=end_idx, X=exog_future, alpha=0.05, dynamic=dynamic
            )
            y_pred = pd.DataFrame({"mean": forecast})

        rename_dict = {
            column: column.replace("mean", "target") for column in y_pred.columns if column.startswith("mean")
        }
        y_pred = y_pred.rename(rename_dict, axis=1)
        return y_pred

    def forecast(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]) -> pd.DataFrame:
        """
        Compute autoregressive predictions from a SARIMAX model.

        Parameters
        ----------
        df:
            Features dataframe
        prediction_interval:
             If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution

        Returns
        -------
        :
            DataFrame with predictions
        """
        return self._make_prediction(df=df, prediction_interval=prediction_interval, quantiles=quantiles, dynamic=True)

    def predict(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]) -> pd.DataFrame:
        """
        Compute predictions from a SARIMAX model and use true in-sample data as lags if possible.

        Parameters
        ----------
        df:
            Features dataframe
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution

        Returns
        -------
        :
            DataFrame with predictions
        """
        return self._make_prediction(df=df, prediction_interval=prediction_interval, quantiles=quantiles, dynamic=False)

    @abstractmethod
    def _get_fit_results(self, endog: pd.Series, exog: pd.DataFrame) -> SARIMAXResultsWrapper:
        pass

    def _check_df(self, df: pd.DataFrame, horizon: Optional[int] = None):
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")
        column_to_drop = [col for col in df.columns if col not in ["target", "timestamp"] + self.regressor_columns]
        if column_to_drop:
            warnings.warn(
                message=f"SARIMAX model does not work with exogenous features (features unknown in future).\n "
                f"{column_to_drop} will be dropped"
            )
        if horizon:
            short_regressors = [regressor for regressor in self.regressor_columns if df[regressor].count() < horizon]
            if short_regressors:
                raise ValueError(
                    f"Regressors {short_regressors} are too short for chosen horizon value.\n "
                    "Try lower horizon value, or drop this regressors."
                )

    def _select_regressors(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.regressor_columns:
            exog_future = df[self.regressor_columns]
            exog_future.index = df["timestamp"]
        else:
            exog_future = None
        return exog_future

    def _encode_categoricals(self, df: pd.DataFrame) -> None:
        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        try:
            df.loc[:, categorical_cols] = df[categorical_cols].astype(int)
        except ValueError:
            raise ValueError(
                f"Categorical columns {categorical_cols} can not been converted to int.\n "
                "Try to encode this columns manually."
            )

    def get_model(self) -> SARIMAXResultsWrapper:
        """Get :py:class:`statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper` that is used inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._fit_results


class _SARIMAXAdapter(_SARIMAXBaseAdapter):
    """
    Class for holding Sarimax model.

    Notes
    -----
    We use SARIMAX [1] model from statsmodels package. Statsmodels package uses `exog` attribute for
    `exogenous regressors` which should be known in future, however we use exogenous for
    additional features what is not known in future, and regressors for features we do know in
    future.

    .. `SARIMAX: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>_`

    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12),
        trend: Optional[str] = "c",
        measurement_error: bool = False,
        time_varying_regression: bool = False,
        mle_regression: bool = True,
        simple_differencing: bool = False,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        hamilton_representation: bool = False,
        concentrate_scale: bool = False,
        trend_offset: float = 1,
        use_exact_diffuse: bool = False,
        dates: Optional[List[datetime]] = None,
        freq: Optional[str] = None,
        missing: str = "none",
        validate_specification: bool = True,
        **kwargs,
    ):
        """
        Init SARIMAX model with given params.

        Parameters
        ----------
        order:
            The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters. `d` must be an integer
            indicating the integration order of the process, while
            `p` and `q` may either be an integers indicating the AR and MA
            orders (so that all lags up to those orders are included) or else
            iterables giving specific AR and / or MA lags to include. Default is
            an AR(1) model: (1,0,0).
        seasonal_order:
            The (P,D,Q,s) order of the seasonal component of the model for the
            AR parameters, differences, MA parameters, and periodicity.
            `D` must be an integer indicating the integration order of the process,
            while `P` and `Q` may either be an integers indicating the AR and MA
            orders (so that all lags up to those orders are included) or else
            iterables giving specific AR and / or MA lags to include. `s` is an
            integer giving the periodicity (number of periods in season), often it
            is 4 for quarterly data or 12 for monthly data. Default is no seasonal
            effect.
        trend:
            Parameter controlling the deterministic trend polynomial :math:`A(t)`.
            Can be specified as a string where 'c' indicates a constant (i.e. a
            degree zero component of the trend polynomial), 't' indicates a
            linear trend with time, and 'ct' is both. Can also be specified as an
            iterable defining the non-zero polynomial exponents to include, in
            increasing order. For example, `[1,1,0,1]` denotes
            :math:`a + bt + ct^3`. Default is to not include a trend component.
        measurement_error:
            Whether or not to assume the endogenous observations `endog` were
            measured with error. Default is False.
        time_varying_regression:
            Used when an explanatory variables, `exog`, are provided provided
            to select whether or not coefficients on the exogenous regressors are
            allowed to vary over time. Default is False.
        mle_regression:
            Whether or not to use estimate the regression coefficients for the
            exogenous variables as part of maximum likelihood estimation or through
            the Kalman filter (i.e. recursive least squares). If
            `time_varying_regression` is True, this must be set to False. Default
            is True.
        simple_differencing:
            Whether or not to use partially conditional maximum likelihood
            estimation. If True, differencing is performed prior to estimation,
            which discards the first :math:`s D + d` initial rows but results in a
            smaller state-space formulation. See the Notes section for important
            details about interpreting results when this option is used. If False,
            the full SARIMAX model is put in state-space form so that all
            datapoints can be used in estimation. Default is False.
        enforce_stationarity:
            Whether or not to transform the AR parameters to enforce stationarity
            in the autoregressive component of the model. Default is True.
        enforce_invertibility:
            Whether or not to transform the MA parameters to enforce invertibility
            in the moving average component of the model. Default is True.
        hamilton_representation:
            Whether or not to use the Hamilton representation of an ARMA process
            (if True) or the Harvey representation (if False). Default is False.
        concentrate_scale:
            Whether or not to concentrate the scale (variance of the error term)
            out of the likelihood. This reduces the number of parameters estimated
            by maximum likelihood by one, but standard errors will then not
            be available for the scale parameter.
        trend_offset:
            The offset at which to start time trend values. Default is 1, so that
            if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
            set when the model created by extending a previous dataset.
        use_exact_diffuse:
            Whether or not to use exact diffuse initialization for non-stationary
            states. Default is False (in which case approximate diffuse
            initialization is used).
        dates:
            If no index is given by `endog` or `exog`, an array-like object of
            datetime objects can be provided.
        freq:
            If no index is given by `endog` or `exog`, the frequency of the
            time-series may be specified here as a Pandas offset or offset string.
        missing:
            Available options are 'none', 'drop', and 'raise'. If 'none', no nan
            checking is done. If 'drop', any observations with nans are dropped.
            If 'raise', an error is raised. Default is 'none'.
        validate_specification:
            If True, validation of hyperparameters is performed.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        self.use_exact_diffuse = use_exact_diffuse
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.validate_specification = validate_specification
        self.kwargs = kwargs
        super().__init__()

    def _get_fit_results(self, endog: pd.Series, exog: pd.DataFrame):
        # make it a numpy array for forgetting about indices, it is necessary for seasonal_prediction_with_confidence
        endog_np = endog.values
        model = SARIMAX(
            endog=endog_np,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            measurement_error=self.measurement_error,
            time_varying_regression=self.time_varying_regression,
            mle_regression=self.mle_regression,
            simple_differencing=self.simple_differencing,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            hamilton_representation=self.hamilton_representation,
            concentrate_scale=self.concentrate_scale,
            trend_offset=self.trend_offset,
            use_exact_diffuse=self.use_exact_diffuse,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
            validate_specification=self.validate_specification,
            **self.kwargs,
        )
        result = model.fit()
        return result


class SARIMAXModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """
    Class for holding Sarimax model.

    Method ``predict`` can use true target values only on train data on future data autoregression
    forecasting will be made even if targets are known.

    Notes
    -----
    We use :py:class:`statsmodels.tsa.sarimax.SARIMAX`. Statsmodels package uses `exog` attribute for
    `exogenous regressors` which should be known in future, however we use exogenous for
    additional features what is not known in future, and regressors for features we do know in
    future.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12),
        trend: Optional[str] = "c",
        measurement_error: bool = False,
        time_varying_regression: bool = False,
        mle_regression: bool = True,
        simple_differencing: bool = False,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        hamilton_representation: bool = False,
        concentrate_scale: bool = False,
        trend_offset: float = 1,
        use_exact_diffuse: bool = False,
        dates: Optional[List[datetime]] = None,
        freq: Optional[str] = None,
        missing: str = "none",
        validate_specification: bool = True,
        **kwargs,
    ):
        """
        Init SARIMAX model with given params.

        Parameters
        ----------
        order:
            The (p,d,q) order of the model for the number of AR parameters,
            differences, and MA parameters. `d` must be an integer
            indicating the integration order of the process, while
            `p` and `q` may either be an integers indicating the AR and MA
            orders (so that all lags up to those orders are included) or else
            iterables giving specific AR and / or MA lags to include. Default is
            an AR(1) model: (1,0,0).
        seasonal_order:
            The (P,D,Q,s) order of the seasonal component of the model for the
            AR parameters, differences, MA parameters, and periodicity.
            `D` must be an integer indicating the integration order of the process,
            while `P` and `Q` may either be an integers indicating the AR and MA
            orders (so that all lags up to those orders are included) or else
            iterables giving specific AR and / or MA lags to include. `s` is an
            integer giving the periodicity (number of periods in season), often it
            is 4 for quarterly data or 12 for monthly data. Default is no seasonal
            effect.
        trend:
            Parameter controlling the deterministic trend polynomial :math:`A(t)`.
            Can be specified as a string where 'c' indicates a constant (i.e. a
            degree zero component of the trend polynomial), 't' indicates a
            linear trend with time, and 'ct' is both. Can also be specified as an
            iterable defining the non-zero polynomial exponents to include, in
            increasing order. For example, `[1,1,0,1]` denotes
            :math:`a + bt + ct^3`. Default is to not include a trend component.
        measurement_error:
            Whether or not to assume the endogenous observations `endog` were
            measured with error. Default is False.
        time_varying_regression:
            Used when an explanatory variables, `exog`, are provided provided
            to select whether or not coefficients on the exogenous regressors are
            allowed to vary over time. Default is False.
        mle_regression:
            Whether or not to use estimate the regression coefficients for the
            exogenous variables as part of maximum likelihood estimation or through
            the Kalman filter (i.e. recursive least squares). If
            `time_varying_regression` is True, this must be set to False. Default
            is True.
        simple_differencing:
            Whether or not to use partially conditional maximum likelihood
            estimation. If True, differencing is performed prior to estimation,
            which discards the first :math:`s D + d` initial rows but results in a
            smaller state-space formulation. See the Notes section for important
            details about interpreting results when this option is used. If False,
            the full SARIMAX model is put in state-space form so that all
            datapoints can be used in estimation. Default is False.
        enforce_stationarity:
            Whether or not to transform the AR parameters to enforce stationarity
            in the autoregressive component of the model. Default is True.
        enforce_invertibility:
            Whether or not to transform the MA parameters to enforce invertibility
            in the moving average component of the model. Default is True.
        hamilton_representation:
            Whether or not to use the Hamilton representation of an ARMA process
            (if True) or the Harvey representation (if False). Default is False.
        concentrate_scale:
            Whether or not to concentrate the scale (variance of the error term)
            out of the likelihood. This reduces the number of parameters estimated
            by maximum likelihood by one, but standard errors will then not
            be available for the scale parameter.
        trend_offset:
            The offset at which to start time trend values. Default is 1, so that
            if `trend='t'` the trend is equal to 1, 2, ..., nobs. Typically is only
            set when the model created by extending a previous dataset.
        use_exact_diffuse:
            Whether or not to use exact diffuse initialization for non-stationary
            states. Default is False (in which case approximate diffuse
            initialization is used).
        dates:
            If no index is given by `endog` or `exog`, an array-like object of
            datetime objects can be provided.
        freq:
            If no index is given by `endog` or `exog`, the frequency of the
            time-series may be specified here as a Pandas offset or offset string.
        missing:
            Available options are 'none', 'drop', and 'raise'. If 'none', no nan
            checking is done. If 'drop', any observations with nans are dropped.
            If 'raise', an error is raised. Default is 'none'.
        validate_specification:
            If True, validation of hyperparameters is performed.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        self.use_exact_diffuse = use_exact_diffuse
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.validate_specification = validate_specification
        self.kwargs = kwargs
        super(SARIMAXModel, self).__init__(
            base_model=_SARIMAXAdapter(
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                measurement_error=self.measurement_error,
                time_varying_regression=self.time_varying_regression,
                mle_regression=self.mle_regression,
                simple_differencing=self.simple_differencing,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
                hamilton_representation=self.hamilton_representation,
                concentrate_scale=self.concentrate_scale,
                trend_offset=self.trend_offset,
                use_exact_diffuse=self.use_exact_diffuse,
                dates=self.dates,
                freq=self.freq,
                missing=self.missing,
                validate_specification=self.validate_specification,
                **self.kwargs,
            )
        )
