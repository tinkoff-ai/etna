import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.libs.pmdarima_utils import seasonal_prediction_with_confidence
from etna.models.base import BaseAdapter
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.utils import determine_freq
from etna.models.utils import determine_num_steps
from etna.models.utils import select_observations

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
        self._last_train_timestamp = None

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
        self._check_not_used_columns(df)

        exog_train = self._select_regressors(df)
        self._fit_results = self._get_fit_results(endog=df["target"], exog=exog_train)

        self._freq = determine_freq(timestamps=df["timestamp"])
        self._first_train_timestamp = df["timestamp"].min()
        self._last_train_timestamp = df["timestamp"].max()

        return self

    def _make_prediction(
        self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float], dynamic: bool
    ) -> pd.DataFrame:
        """Make predictions taking into account ``dynamic`` parameter."""
        if self._fit_results is None:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

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

    def _check_not_used_columns(self, df: pd.DataFrame):
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")

        columns_not_used = [col for col in df.columns if col not in ["target", "timestamp"] + self.regressor_columns]
        if columns_not_used:
            warnings.warn(
                message=f"This model doesn't work with exogenous features unknown in future. "
                f"Columns {columns_not_used} won't be used."
            )

    def _select_regressors(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Select data with regressors.

        During fit there can't be regressors with NaNs, they are removed at higher level.
        Look at the issue: https://github.com/tinkoff-ai/etna/issues/557

        During prediction without validation NaNs in regressors lead to exception from the underlying model.

        This model requires data to be in numeric dtype, but doesn't support boolean, so it was decided to use float.
        """
        if self.regressor_columns is None:
            raise ValueError("Something went wrong, regressor_columns is None!")

        regressors_with_nans = [regressor for regressor in self.regressor_columns if df[regressor].isna().sum() > 0]
        if regressors_with_nans:
            raise ValueError(
                f"Regressors {regressors_with_nans} contain NaN values. "
                "Try to lower horizon value, or drop these regressors."
            )

        if self.regressor_columns:
            try:
                result = df[self.regressor_columns].astype(float)
            except ValueError as e:
                raise ValueError(f"Only convertible to float features are allowed! Error: {str(e)}")
            result.index = df["timestamp"]
        else:
            result = None

        return result

    def get_model(self) -> SARIMAXResultsWrapper:
        """Get :py:class:`statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper` that is used inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._fit_results

    @staticmethod
    def _prepare_components_df(components: np.ndarray, model: SARIMAX) -> pd.DataFrame:
        """Prepare `pd.DataFrame` with components."""
        if model.exog_names is not None:
            components_names = model.exog_names[:]
        else:
            components_names = []

        if model.seasonal_periods == 0:
            components_names.append("arima")
        else:
            components_names.append("sarima")

        df = pd.DataFrame(data=components, columns=components_names)
        return df.add_prefix("target_component_")

    @staticmethod
    def _prepare_design_matrix(ssm: SimulationSmoother) -> np.ndarray:
        """Extract design matrix from state space model."""
        design_mat = ssm["design"]
        if len(design_mat.shape) == 2:
            design_mat = design_mat[..., np.newaxis]

        return design_mat

    def _mle_regression_decomposition(self, state: np.ndarray, ssm: SimulationSmoother, exog: np.ndarray) -> np.ndarray:
        """Estimate SARIMAX components for MLE regression case.

        SARIMAX representation as SSM: https://www.statsmodels.org/dev/statespace.html
        In MLE case exogenous data fitted separately from other components:
        https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/sarimax.py#L1644
        """
        # get design matrix from SSM
        design_mat = self._prepare_design_matrix(ssm)

        # estimate SARIMA component
        components = np.sum(design_mat * state, axis=1).T

        if len(exog) > 0:
            # restore parameters for exogenous variabales
            exog_params = np.linalg.lstsq(a=exog, b=np.squeeze(ssm["obs_intercept"]))[0]

            # estimate exogenous components and append to others
            weighted_exog = exog * exog_params[np.newaxis]
            components = np.concatenate([weighted_exog, components], axis=1)

        return components

    def _state_regression_decomposition(self, state: np.ndarray, ssm: SimulationSmoother, k_exog: int) -> np.ndarray:
        """Estimate SARIMAX components for state regression case.

        SARIMAX representation as SSM: https://www.statsmodels.org/dev/statespace.html
        In state regression case parameters for exogenous variables estimated inside SSM.
        """
        # get design matrix from SSM
        design_mat = self._prepare_design_matrix(ssm)

        if k_exog > 0:
            # estimate SARIMA component
            sarima = np.sum(design_mat[:, :-k_exog] * state[:-k_exog], axis=1)

            # obtain params from SSM and estimate exogenous components
            weighted_exog = np.squeeze(design_mat[:, -k_exog:] * state[-k_exog:])
            components = np.concatenate([weighted_exog, sarima], axis=0).T

        else:
            # in this case we can take whole matrix for SARIMA component
            components = np.sum(design_mat * state, axis=1).T

        return components

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate prediction components.

        Parameters
        ----------
        df:
            features dataframe

        Returns
        -------
        :
            dataframe with prediction components
        """
        if self._fit_results is None:
            raise ValueError("Model is not fitted! Fit the model before estimating forecast components!")

        if self._last_train_timestamp < df["timestamp"].max() or self._first_train_timestamp > df["timestamp"].min():
            raise NotImplementedError(
                "This model can't make prediction decomposition on future out-of-sample data! "
                "Use method forecast for future out-of-sample prediction decomposition."
            )

        fit_results = self._fit_results
        model = fit_results.model

        if model.hamilton_representation:
            raise NotImplementedError(
                "Prediction decomposition is not implemented for Hamilton representation of ARMA!"
            )

        state = fit_results.predicted_state[:, :-1]

        if model.mle_regression:
            components = self._mle_regression_decomposition(state=state, ssm=model.ssm, exog=model.exog)

        else:
            components = self._state_regression_decomposition(state=state, ssm=model.ssm, k_exog=model.k_exog)

        components_df = self._prepare_components_df(components=components, model=model)

        components_df = select_observations(
            df=components_df,
            timestamps=df["timestamp"],
            start=self._first_train_timestamp,
            end=self._last_train_timestamp,
            freq=self._freq,
        )

        return components_df

    def forecast_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate forecast components.

        Parameters
        ----------
        df:
            features dataframe

        Returns
        -------
        :
            dataframe with forecast components
        """
        if self._fit_results is None:
            raise ValueError("Model is not fitted! Fit the model before estimating forecast components!")

        start_timestamp = df["timestamp"].min()
        end_timestamp = df["timestamp"].max()

        if start_timestamp < self._last_train_timestamp:
            raise NotImplementedError(
                "This model can't make forecast decomposition on history data! "
                "Use method predict for in-sample prediction decomposition."
            )

        # determine index of start_timestamp if counting from last timestamp of train
        start_idx = determine_num_steps(
            start_timestamp=self._last_train_timestamp, end_timestamp=start_timestamp, freq=self._freq  # type: ignore
        )
        # determine index of end_timestamp if counting from last timestamp of train
        end_idx = determine_num_steps(
            start_timestamp=self._last_train_timestamp, end_timestamp=end_timestamp, freq=self._freq  # type: ignore
        )

        if start_idx > 1:
            raise NotImplementedError(
                "This model can't make forecast decomposition on out-of-sample data that goes after training data with a gap! "
                "You can only forecast from the next point after the last one in the training dataset."
            )

        horizon = end_idx
        fit_results = self._fit_results

        model = fit_results.model
        if model.hamilton_representation:
            raise NotImplementedError(
                "Prediction decomposition is not implemented for Hamilton representation of ARMA!"
            )

        exog_future = self._select_regressors(df)

        forecast_results = fit_results.get_forecast(horizon, exog=exog_future).prediction_results.results
        state = forecast_results.predicted_state[:, :-1]

        if model.mle_regression:
            # If there are no exog variales `mle_regression` will be set to `False`
            # even if user set to `True`.
            components = self._mle_regression_decomposition(
                state=state, ssm=forecast_results.model, exog=exog_future.values  # type: ignore
            )

        else:
            components = self._state_regression_decomposition(
                state=state, ssm=forecast_results.model, k_exog=model.k_exog
            )

        components_df = self._prepare_components_df(components=components, model=model)

        components_df = select_observations(
            df=components_df, timestamps=df["timestamp"], end=df["timestamp"].max(), periods=horizon, freq=self._freq
        )

        return components_df


class _SARIMAXAdapter(_SARIMAXBaseAdapter):
    """
    Class for holding SARIMAX model.

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
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
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
        **kwargs:
            Additional parameters for :py:class:`statsmodels.tsa.sarimax.SARIMAX`.
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
    Class for holding SARIMAX model.

    Method ``predict`` can use true target values only on train data on future data autoregression
    forecasting will be made even if targets are known.

    Notes
    -----
    We use :py:class:`statsmodels.tsa.sarimax.SARIMAX`. Statsmodels package uses `exog` attribute for
    `exogenous regressors` which should be known in future, however we use exogenous for
    additional features what is not known in future, and regressors for features we do know in
    future.

    This model supports in-sample and out-of-sample prediction decomposition.
    Prediction components for SARIMAX model are: exogenous and SARIMA components.
    Decomposition is obtained directly from fitted model parameters.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
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
        **kwargs:
            Additional parameters for :py:class:`statsmodels.tsa.sarimax.SARIMAX`.
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

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``order.0``, ``order.1``, ``order.2``, ``trend``.
        If ``self.num_periods`` is greater than zero, then it also tunes parameters:
        ``seasonal_order.0``, ``seasonal_order.1``, ``seasonal_order.2``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        grid: Dict[str, "BaseDistribution"] = {
            "order.0": IntDistribution(low=1, high=6),
            "order.1": IntDistribution(low=1, high=2),
            "order.2": IntDistribution(low=1, high=6),
            "trend": CategoricalDistribution(["n", "c", "t", "ct"]),
        }

        num_periods = self.seasonal_order[3]
        if num_periods > 0:
            grid.update(
                {
                    "seasonal_order.0": IntDistribution(low=0, high=2),
                    "seasonal_order.1": IntDistribution(low=0, high=1),
                    "seasonal_order.2": IntDistribution(low=0, high=1),
                }
            )

        return grid
