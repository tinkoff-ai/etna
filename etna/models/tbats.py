from typing import Iterable
from typing import Optional
from typing import Tuple
from warnings import warn

import numpy as np
import pandas as pd
from tbats.abstract import ContextInterface
from tbats.abstract import Estimator
from tbats.bats import BATS
from tbats.tbats import TBATS
from tbats.tbats.Model import Model

from etna.models.base import BaseAdapter
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.utils import determine_freq
from etna.models.utils import determine_num_steps
from etna.models.utils import select_observations


class _TBATSAdapter(BaseAdapter):
    def __init__(self, model: Estimator):
        self._model = model
        self._fitted_model: Optional[Model] = None
        self._first_train_timestamp = None
        self._last_train_timestamp = None
        self._freq: Optional[str] = None

    def _check_not_used_columns(self, df: pd.DataFrame):
        columns = df.columns
        columns_not_used = set(columns).difference({"target", "timestamp"})
        if columns_not_used:
            warn(
                message=f"This model doesn't work with exogenous features. "
                f"Columns {columns_not_used} won't be used."
            )

    def fit(self, df: pd.DataFrame, regressors: Iterable[str]):
        self._freq = determine_freq(timestamps=df["timestamp"])
        self._check_not_used_columns(df)

        target = df["target"]
        self._fitted_model = self._model.fit(target)
        self._first_train_timestamp = df["timestamp"].min()
        self._last_train_timestamp = df["timestamp"].max()

        return self

    def forecast(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Iterable[float]) -> pd.DataFrame:
        if self._fitted_model is None or self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        steps_to_forecast = self._get_steps_to_forecast(df=df)
        steps_to_skip = steps_to_forecast - df.shape[0]

        y_pred = pd.DataFrame()
        if prediction_interval:
            for quantile in quantiles:
                pred, confidence_intervals = self._fitted_model.forecast(
                    steps=steps_to_forecast, confidence_level=quantile
                )
                y_pred["target"] = pred
                if quantile < 1 / 2:
                    y_pred[f"target_{quantile:.4g}"] = confidence_intervals["lower_bound"]
                else:
                    y_pred[f"target_{quantile:.4g}"] = confidence_intervals["upper_bound"]
        else:
            pred = self._fitted_model.forecast(steps=steps_to_forecast)
            y_pred["target"] = pred

        # skip non-relevant timestamps
        y_pred = y_pred.iloc[steps_to_skip:].reset_index(drop=True)

        return y_pred

    def predict(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Iterable[float]) -> pd.DataFrame:
        if self._fitted_model is None or self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        train_timestamp = pd.date_range(
            start=str(self._first_train_timestamp), end=str(self._last_train_timestamp), freq=self._freq
        )

        if not (set(df["timestamp"]) <= set(train_timestamp)):
            raise NotImplementedError(
                "This model can't make predict on future out-of-sample data! "
                "Use forecast method for this type of prediction."
            )

        y_pred = pd.DataFrame()
        y_pred["target"] = self._fitted_model.y_hat
        y_pred["timestamp"] = train_timestamp

        if prediction_interval:
            for quantile in quantiles:
                confidence_intervals = self._fitted_model._calculate_confidence_intervals(
                    y_pred["target"].values, quantile
                )

                if quantile < 1 / 2:
                    y_pred[f"target_{quantile:.4g}"] = confidence_intervals["lower_bound"]
                else:
                    y_pred[f"target_{quantile:.4g}"] = confidence_intervals["upper_bound"]

        # selecting time points from provided dataframe
        y_pred.set_index("timestamp", inplace=True)
        y_pred = y_pred.loc[df["timestamp"]]
        y_pred.reset_index(drop=True, inplace=True)

        return y_pred

    def get_model(self) -> Model:
        """Get internal :py:class:`tbats.tbats.Model` model that was fitted inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._fitted_model

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
        if self._fitted_model is None or self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before estimating forecast components!")

        if df["timestamp"].min() <= self._last_train_timestamp:
            raise NotImplementedError(
                "This model can't make forecast decomposition on history data! "
                "Use method predict for in-sample prediction decomposition."
            )

        self._check_components()

        horizon = self._get_steps_to_forecast(df=df)
        raw_components = self._decompose_forecast(horizon=horizon)
        components = self._process_components(raw_components=raw_components)

        components = select_observations(
            df=components, timestamps=df["timestamp"], end=df["timestamp"].max(), periods=horizon, freq=self._freq
        )

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
        if self._fitted_model is None or self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before estimating forecast components!")

        if self._last_train_timestamp < df["timestamp"].max() or self._first_train_timestamp > df["timestamp"].min():
            raise NotImplementedError(
                "This model can't make prediction decomposition on future out-of-sample data! "
                "Use method forecast for future out-of-sample prediction decomposition."
            )

        self._check_components()

        raw_components = self._decompose_predict()
        components = self._process_components(raw_components=raw_components)

        components = select_observations(
            df=components,
            timestamps=df["timestamp"],
            start=self._first_train_timestamp,
            end=self._last_train_timestamp,
            freq=self._freq,
        )

        return components

    def _get_steps_to_forecast(self, df: pd.DataFrame) -> int:
        if self._freq is None:
            raise ValueError("Data frequency is not set!")

        if df["timestamp"].min() <= self._last_train_timestamp:
            raise NotImplementedError(
                "This model can't make forecast on history data! Use method predict for in-sample prediction."
            )

        steps_to_forecast = determine_num_steps(
            start_timestamp=self._last_train_timestamp, end_timestamp=df["timestamp"].max(), freq=self._freq
        )
        return steps_to_forecast

    def _check_components(self):
        """Compare fitted model params with the initial params.

        TBATS tries different models and selects best based on AIC.
        That's why some components may not be present in fitted model.
        """
        if self._fitted_model is None:
            raise ValueError("Fitted model is not set!")

        fitted_model_params = self._fitted_model.params.components

        not_fitted_components = []
        seasonal_periods = self._model.seasonal_periods
        if (
            seasonal_periods is not None
            and len(seasonal_periods) > 0
            and len(fitted_model_params.seasonal_periods) == 0
        ):
            not_fitted_components.append("Seasonal")

        if self._model.use_arma_errors and not fitted_model_params.use_arma_errors:
            not_fitted_components.append("ARMA")

        if len(not_fitted_components) > 0:
            warn(f"Following components are not fitted: {', '.join(not_fitted_components)}!")

    def _rescale_components(self, raw_components: np.ndarray) -> np.ndarray:
        """Rescale components when Box-Cox transform used."""
        if self._fitted_model is None:
            raise ValueError("Fitted model is not set!")

        transformed_pred = np.sum(raw_components, axis=1)
        pred = self._fitted_model._inv_boxcox(transformed_pred)
        components = raw_components * pred[..., np.newaxis] / transformed_pred[..., np.newaxis]
        return components

    def _decompose_forecast(self, horizon: int) -> np.ndarray:
        """Estimate raw forecast components."""
        if self._fitted_model is None:
            raise ValueError("Fitted model is not set!")

        model = self._fitted_model
        state_matrix = model.matrix.make_F_matrix()
        component_weights = model.matrix.make_w_vector()

        state = model.x_last
        components = []
        for _ in range(horizon):
            components.append(component_weights * state)
            state = state_matrix @ state

        raw_components = np.stack(components, axis=0)

        if model.params.components.use_box_cox:
            raw_components = self._rescale_components(raw_components)

        return raw_components

    def _decompose_predict(self) -> np.ndarray:
        """Estimate raw prediction components."""
        if self._fitted_model is None:
            raise ValueError("Fitted model is not set!")

        model = self._fitted_model
        state_matrix = model.matrix.make_F_matrix()
        component_weights = model.matrix.make_w_vector()
        error_weights = model.matrix.make_g_vector()

        steps = len(model.y)
        state = model.params.x0
        weighted_error = model.resid_boxcox[..., np.newaxis] * error_weights[np.newaxis]

        components = []
        for t in range(steps):
            components.append(component_weights * state)
            state = state_matrix @ state + weighted_error[t]

        raw_components = np.stack(components, axis=0)

        if model.params.components.use_box_cox:
            raw_components = self._rescale_components(raw_components)

        return raw_components

    def _process_components(self, raw_components: np.ndarray) -> pd.DataFrame:
        """Select meaningful components and assign names to them."""
        if self._fitted_model is None:
            raise ValueError("Fitted model is not set!")

        params_components = self._fitted_model.params.components
        named_components = dict()

        named_components["local_level"] = raw_components[:, 0]

        component_idx = 1
        if params_components.use_trend:
            named_components["trend"] = raw_components[:, component_idx]
            component_idx += 1

        if len(params_components.seasonal_periods) != 0:
            seasonal_periods = params_components.seasonal_periods

            if hasattr(params_components, "seasonal_harmonics"):
                # TBATS
                seasonal_harmonics = params_components.seasonal_harmonics
                for seasonal_period, seasonal_harmonic in zip(seasonal_periods, seasonal_harmonics):
                    named_components[f"seasonal(s={seasonal_period})"] = np.sum(
                        raw_components[:, component_idx : component_idx + 2 * seasonal_harmonic], axis=1
                    )
                    component_idx += 2 * seasonal_harmonic

            else:
                # BATS
                component_idx -= 1
                for seasonal_period in seasonal_periods:
                    component_idx += seasonal_period
                    named_components[f"seasonal(s={seasonal_period})"] = raw_components[:, component_idx]

            component_idx += 1

        if params_components.p > 0 or params_components.q > 0:
            p, q = params_components.p, params_components.q
            named_components[f"arma(p={p},q={q})"] = np.sum(
                raw_components[:, component_idx : component_idx + p + q], axis=1
            )

        return pd.DataFrame(data=named_components).add_prefix("target_component_")


class BATSModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """Class for holding segment interval BATS model.

    Notes
    -----
    This model supports in-sample and out-of-sample prediction decomposition.
    Prediction components for BATS model are: local level, trend, seasonality and ARMA component.
    In-sample and out-of-sample decompositions components are estimated directly from the fitted model parameters.
    Box-Cox transform supported with components proportional rescaling.
    """

    def __init__(
        self,
        use_box_cox: Optional[bool] = None,
        box_cox_bounds: Tuple[int, int] = (0, 1),
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        seasonal_periods: Optional[Iterable[int]] = None,
        use_arma_errors: bool = True,
        show_warnings: bool = True,
        n_jobs: Optional[int] = None,
        multiprocessing_start_method: str = "spawn",
        context: Optional[ContextInterface] = None,
    ):
        """Create BATSModel with given parameters.

        Parameters
        ----------
        use_box_cox: bool or None, optional (default=None)
            If Box-Cox transformation of original series should be applied.
            When None both cases shall be considered and better is selected by AIC.
        box_cox_bounds: tuple, shape=(2,), optional (default=(0, 1))
            Minimal and maximal Box-Cox parameter values.
        use_trend: bool or None, optional (default=None)
            Indicates whether to include a trend or not.
            When None both cases shall be considered and better is selected by AIC.
        use_damped_trend: bool or None, optional (default=None)
            Indicates whether to include a damping parameter in the trend or not.
            Applies only when trend is used.
            When None both cases shall be considered and better is selected by AIC.
        seasonal_periods: iterable or array-like of int values, optional (default=None)
            Length of each of the periods (amount of observations in each period).
            BATS accepts only int values here.
            When None or empty array, non-seasonal model shall be fitted.
        use_arma_errors: bool, optional (default=True)
            When True BATS will try to improve the model by modelling residuals with ARMA.
            Best model will be selected by AIC.
            If False, ARMA residuals modeling will not be considered.
        show_warnings: bool, optional (default=True)
            If warnings should be shown or not.
            Also see Model.warnings variable that contains all model related warnings.
        n_jobs: int, optional (default=None)
            How many jobs to run in parallel when fitting BATS model.
            When not provided BATS shall try to utilize all available cpu cores.
        multiprocessing_start_method: str, optional (default='spawn')
            How threads should be started.
            See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        context: abstract.ContextInterface, optional (default=None)
            For advanced users only. Provide this to override default behaviors
        """
        self.model = BATS(
            use_box_cox=use_box_cox,
            box_cox_bounds=box_cox_bounds,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors,
            show_warnings=show_warnings,
            n_jobs=n_jobs,
            multiprocessing_start_method=multiprocessing_start_method,
            context=context,
        )
        super().__init__(base_model=_TBATSAdapter(self.model))


class TBATSModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """Class for holding segment interval TBATS model.

    Notes
    -----
    This model supports in-sample and out-of-sample prediction decomposition.
    Prediction components for TBATS model are: local level, trend, seasonality and ARMA component.
    In-sample and out-of-sample decompositions components are estimated directly from the fitted model parameters.
    Box-Cox transform supported with components proportional rescaling.
    """

    def __init__(
        self,
        use_box_cox: Optional[bool] = None,
        box_cox_bounds: Tuple[int, int] = (0, 1),
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        seasonal_periods: Optional[Iterable[int]] = None,
        use_arma_errors: bool = True,
        show_warnings: bool = True,
        n_jobs: Optional[int] = None,
        multiprocessing_start_method: str = "spawn",
        context: Optional[ContextInterface] = None,
    ):
        """Create TBATSModel with given parameters.

        Parameters
        ----------
        use_box_cox: bool or None, optional (default=None)
            If Box-Cox transformation of original series should be applied.
            When None both cases shall be considered and better is selected by AIC.
        box_cox_bounds: tuple, shape=(2,), optional (default=(0, 1))
            Minimal and maximal Box-Cox parameter values.
        use_trend: bool or None, optional (default=None)
            Indicates whether to include a trend or not.
            When None both cases shall be considered and better is selected by AIC.
        use_damped_trend: bool or None, optional (default=None)
            Indicates whether to include a damping parameter in the trend or not.
            Applies only when trend is used.
            When None both cases shall be considered and better is selected by AIC.
        seasonal_periods: iterable or array-like of floats, optional (default=None)
            Length of each of the periods (amount of observations in each period).
            TBATS accepts int and float values here.
            When None or empty array, non-seasonal model shall be fitted.
        use_arma_errors: bool, optional (default=True)
            When True BATS will try to improve the model by modelling residuals with ARMA.
            Best model will be selected by AIC.
            If False, ARMA residuals modeling will not be considered.
        show_warnings: bool, optional (default=True)
            If warnings should be shown or not.
            Also see Model.warnings variable that contains all model related warnings.
        n_jobs: int, optional (default=None)
            How many jobs to run in parallel when fitting BATS model.
            When not provided BATS shall try to utilize all available cpu cores.
        multiprocessing_start_method: str, optional (default='spawn')
            How threads should be started.
            See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        context: abstract.ContextInterface, optional (default=None)
            For advanced users only. Provide this to override default behaviors
        """
        self.model = TBATS(
            use_box_cox=use_box_cox,
            box_cox_bounds=box_cox_bounds,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors,
            show_warnings=show_warnings,
            n_jobs=n_jobs,
            multiprocessing_start_method=multiprocessing_start_method,
            context=context,
        )
        super().__init__(base_model=_TBATSAdapter(self.model))
