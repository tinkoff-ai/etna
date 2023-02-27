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
from etna.models.utils import determine_num_steps


class _TBATSAdapter(BaseAdapter):
    def __init__(self, model: Estimator):
        self._model = model
        self._fitted_model: Optional[Model] = None
        self._last_train_timestamp = None
        self._freq = None

    def fit(self, df: pd.DataFrame, regressors: Iterable[str]):
        freq = pd.infer_freq(df["timestamp"], warn=False)
        if freq is None:
            raise ValueError("Can't determine frequency of a given dataframe")

        target = df["target"]
        self._fitted_model = self._model.fit(target)
        self._last_train_timestamp = df["timestamp"].max()
        self._freq = freq

        return self

    def forecast(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Iterable[float]) -> pd.DataFrame:
        if self._fitted_model is None or self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before calling predict method!")

        if df["timestamp"].min() <= self._last_train_timestamp:
            raise NotImplementedError(
                "It is not possible to make in-sample predictions with BATS/TBATS model! "
                "In-sample predictions aren't supported by current implementation."
            )

        steps_to_forecast = determine_num_steps(
            start_timestamp=self._last_train_timestamp, end_timestamp=df["timestamp"].max(), freq=self._freq
        )
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
        raise NotImplementedError("Method predict isn't currently implemented!")

    def get_model(self) -> Model:
        """Get internal :py:class:`tbats.tbats.Model` model that was fitted inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._fitted_model

    def forecast_components(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Estimate forecast components.

        Parameters
        ----------
        df:
            data for estimation forecast components
        horizon:
            forecast horizon

        Returns
        -------
        :
            dataframe with forecast components
        """
        if self._fitted_model is None or self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before estimating forecast components!")

        self._check_components()
        raw_components = self._decompose_forecast(horizon=horizon)
        components = self._named_components(raw_components=raw_components)

        return pd.DataFrame(data=components)

    def predict_components(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Estimate prediction components.

        Parameters
        ----------
        df:
            data for estimation prediction components
        horizon:
            forecast horizon

        Returns
        -------
        :
            dataframe with prediction components
        """
        raise NotImplementedError("Prediction decomposition isn't currently implemented!")

    def _check_components(self):
        """Compare fitted model params with the initial params."""
        fitted_model_params = self._fitted_model.params.components

        seasonal_periods = self._model.seasonal_periods
        if (
            seasonal_periods is not None
            and len(seasonal_periods) > 0
            and len(fitted_model_params.seasonal_periods) == 0
        ):
            warn("Seasonal components is not fitted!")

        if self._model.use_arma_errors and not fitted_model_params.use_arma_errors:
            warn("ARMA components is not fitted!")

        if self._model.use_box_cox and not fitted_model_params.use_box_cox:
            warn("Box-Cox transform is not fitted!")

        if self._model.use_trend and not fitted_model_params.use_trend:
            warn("Trend is not fitted!")

    def _decompose_forecast(self, horizon):
        """Estimate raw forecast components."""
        model = self._fitted_model
        state_matrix = model.matrix.make_F_matrix()
        component_weights = model.matrix.make_w_vector()

        state = model.x_last
        components = []
        for _ in range(horizon):
            components.append(component_weights * state)
            state = state_matrix @ state

        components = np.stack(components, axis=0)

        if model.params.components.use_box_cox:
            transformed_pred = np.sum(components, axis=1)
            pred = model._inv_boxcox(transformed_pred)
            components = components * pred[..., np.newaxis] / transformed_pred[..., np.newaxis]

        return components

    def _named_components(self, raw_components):
        """Prepare components with names."""
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
                seasonal_harmonics = params_components.seasonal_harmonics
                for seasonal_period, seasonal_harmonic in zip(seasonal_periods, seasonal_harmonics):
                    named_components[f"seasonal(s={seasonal_period})"] = np.sum(
                        raw_components[:, component_idx : component_idx + 2 * seasonal_harmonic], axis=1
                    )
                    component_idx += 2 * seasonal_harmonic

            else:
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

        return named_components


class BATSModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """Class for holding segment interval BATS model."""

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
    """Class for holding segment interval TBATS model."""

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
