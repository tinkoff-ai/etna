import warnings
from datetime import datetime
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from etna.datasets import TSDataset
from etna.models.base import PerSegmentModel
from etna.models.base import log_decorator

warnings.filterwarnings(
    message="No frequency information was provided, so inferred frequency .* will be used",
    action="ignore",
    category=ValueWarning,
    module="statsmodels.tsa.base.tsa_model",
)


class _SARIMAXModel:
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
        self._model: Optional[SARIMAX] = None
        self._result: Optional[SARIMAX] = None

    def fit(self, df: pd.DataFrame) -> "_SARIMAXModel":
        """
        Fits a SARIMAX model.

        Parameters
        ----------
        df:
            Features dataframe

        Returns
        -------
        self: SARIMAX
            fitted model
        """
        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        try:
            df.loc[:, categorical_cols] = df[categorical_cols].astype(int)
        except ValueError:
            raise ValueError(
                f"Categorical columns {categorical_cols} can not been converted to int.\n "
                "Try to encode this columns manually."
            )

        self._check_df(df)

        targets = df["target"]
        targets.index = df["timestamp"]

        exog_train = self._select_regressors(df)
        regressor_columns = None
        if exog_train is not None:
            regressor_columns = exog_train.columns.values

        if regressor_columns:
            addition_to_params = len(regressor_columns) * [0]
        else:
            addition_to_params = []

        self._model = SARIMAX(
            endog=targets,
            exog=exog_train,
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
        # expect every params but last to be near 0
        start_params = [0, 0, 0, 0] + addition_to_params + [1]
        self._result = self._model.fit(start_params=start_params, disp=False)
        return self

    def predict(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]) -> pd.DataFrame:
        """
        Compute predictions from a SARIMAX model.

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
        y_pred: pd.DataFrame
            DataFrame with predictions
        """
        horizon = len(df)
        self._check_df(df, horizon)

        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        try:
            df.loc[:, categorical_cols] = df[categorical_cols].astype(int)
        except ValueError:
            raise ValueError(
                f"Categorical columns {categorical_cols} can not been converted to int.\n "
                "Try to encode this columns manually."
            )

        exog_future = self._select_regressors(df)
        if prediction_interval:
            forecast = self._result.get_prediction(
                start=df["timestamp"].min(), end=df["timestamp"].max(), dynamic=False, exog=exog_future
            )
            y_pred = pd.DataFrame(forecast.predicted_mean)
            y_pred.rename(
                {"predicted_mean": "mean"},
            )
            for quantile in quantiles:
                # set alpha in the way to get a desirable quantile
                alpha = min(quantile * 2, (1 - quantile) * 2)
                borders = forecast.conf_int(alpha=alpha)
                if quantile < 1 / 2:
                    series = borders["lower target"]
                else:
                    series = borders["upper target"]
                y_pred[f"mean_{quantile:.4g}"] = series
        else:
            forecast = self._result.get_prediction(
                start=df["timestamp"].min(), end=df["timestamp"].max(), dynamic=True, exog=exog_future
            )
            y_pred = pd.DataFrame(forecast.predicted_mean)
            y_pred.rename({"predicted_mean": "mean"}, axis=1, inplace=True)
        return y_pred.reset_index(drop=True, inplace=False)

    def _check_df(self, df: pd.DataFrame, horizon: Optional[int] = None):
        column_to_drop = [
            col for col in df.columns if not col.startswith("regressor") and col not in ["target", "timestamp"]
        ]
        regressor_columns = [col for col in df.columns if col.startswith("regressor")]
        if column_to_drop:
            warnings.warn(
                message=f"SARIMAX model does not work with exogenous features (features unknown in future).\n "
                f"{column_to_drop} will be dropped"
            )
        if horizon:
            short_regressors = [regressor for regressor in regressor_columns if df[regressor].count() < horizon]
            if short_regressors:
                raise ValueError(
                    f"Regressors {short_regressors} are too short for chosen horizon value.\n "
                    "Try lower horizon value, or drop this regressors."
                )

    def _select_regressors(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        regressor_columns = [col for col in df.columns if col.startswith("regressor")]
        if regressor_columns:
            exog_future = df[regressor_columns]
            exog_future.index = df["timestamp"]
        else:
            exog_future = None
        return exog_future


class SARIMAXModel(PerSegmentModel):
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
        super(SARIMAXModel, self).__init__(
            base_model=_SARIMAXModel(
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

    @staticmethod
    def _forecast_one_segment(
        model,
        segment: Union[str, List[str]],
        ts: TSDataset,
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> pd.DataFrame:
        segment_features = ts[:, segment, :]
        segment_features = segment_features.droplevel("segment", axis=1)
        segment_features = segment_features.reset_index()
        dates = segment_features["timestamp"]
        dates.reset_index(drop=True, inplace=True)
        segment_predict = model.predict(
            df=segment_features, prediction_interval=prediction_interval, quantiles=quantiles
        )
        rename_dict = {
            column: column.replace("mean", "target") for column in segment_predict.columns if column.startswith("mean")
        }
        segment_predict = segment_predict.rename(rename_dict, axis=1)
        segment_predict["segment"] = segment
        segment_predict["timestamp"] = dates
        return segment_predict

    @log_decorator
    def forecast(
        self, ts: TSDataset, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)
    ) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataframe with features
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval

        Returns
        -------
        pd.DataFrame
            Models result
        """
        if self._segments is None:
            raise ValueError("The model is not fitted yet, use fit() to train it")

        result_list = list()
        for segment in self._segments:
            model = self._models[segment]

            segment_predict = self._forecast_one_segment(model, segment, ts, prediction_interval, quantiles)
            result_list.append(segment_predict)

        # need real case to test
        result_df = pd.concat(result_list, ignore_index=True)
        result_df = result_df.set_index(["timestamp", "segment"])
        df = ts.to_pandas(flatten=True)
        df = df.set_index(["timestamp", "segment"])
        df = df.combine_first(result_df).reset_index()

        df = TSDataset.to_dataset(df)
        ts.df = df
        ts.inverse_transform()
        return ts
