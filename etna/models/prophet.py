import warnings
from copy import deepcopy
from datetime import datetime
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import pandas as pd

from etna import SETTINGS
from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import FloatDistribution
from etna.models.base import BaseAdapter
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin

if SETTINGS.prophet_required:
    from prophet import Prophet
    from prophet.serialize import model_from_dict
    from prophet.serialize import model_to_dict


class _ProphetAdapter(BaseAdapter):
    """Class for holding Prophet model."""

    predefined_regressors_names = ("floor", "cap")

    def __init__(
        self,
        growth: str = "linear",
        changepoints: Optional[List[datetime]] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Union[str, bool] = "auto",
        weekly_seasonality: Union[str, bool] = "auto",
        daily_seasonality: Union[str, bool] = "auto",
        holidays: Optional[pd.DataFrame] = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.8,
        uncertainty_samples: Union[int, bool] = 1000,
        stan_backend: Optional[str] = None,
        additional_seasonality_params: Iterable[Dict[str, Union[str, float, int]]] = (),
    ):

        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoints = changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        self.additional_seasonality_params = additional_seasonality_params

        self.model = self._create_model()

        self.regressor_columns: Optional[List[str]] = None

    def _create_model(self) -> "Prophet":
        model = Prophet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend,
        )

        for seasonality_params in self.additional_seasonality_params:
            model.add_seasonality(**seasonality_params)

        return model

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

        This model requires data to be in numeric dtype.
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
                result = df[self.regressor_columns].apply(pd.to_numeric)
            except ValueError as e:
                raise ValueError(f"Only convertible to numeric features are allowed! Error: {str(e)}")
        else:
            result = None

        return result

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_ProphetAdapter":
        """
        Fits a Prophet model.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors
        """
        self.regressor_columns = regressors
        self._check_not_used_columns(df)

        prophet_df = self._prepare_prophet_df(df=df)
        for regressor in self.regressor_columns:
            if regressor not in self.predefined_regressors_names:
                self.model.add_regressor(regressor)
        self.model.fit(prophet_df)
        return self

    def predict(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]) -> pd.DataFrame:
        """
        Compute predictions from a Prophet model.

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
        prophet_df = self._prepare_prophet_df(df=df)
        forecast = self.model.predict(prophet_df)
        y_pred = pd.DataFrame(forecast["yhat"])
        if prediction_interval:
            sim_values = self.model.predictive_samples(prophet_df)
            for quantile in quantiles:
                percentile = quantile * 100
                y_pred[f"yhat_{quantile:.4g}"] = self.model.percentile(sim_values["yhat"], percentile, axis=1)
        rename_dict = {
            column: column.replace("yhat", "target") for column in y_pred.columns if column.startswith("yhat")
        }
        y_pred = y_pred.rename(rename_dict, axis=1)
        return y_pred

    def _prepare_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for fit and predict."""
        if self.regressor_columns is None:
            raise ValueError("List of regressor is not set!")

        df = df.reset_index()

        prophet_df = pd.DataFrame()
        prophet_df["y"] = df["target"]
        prophet_df["ds"] = df["timestamp"]

        regressors_data = self._select_regressors(df)
        if regressors_data is not None:
            prophet_df[self.regressor_columns] = regressors_data[self.regressor_columns]

        return prophet_df

    @staticmethod
    def _filter_aggregated_components(components: Iterable[str]) -> Set[str]:
        """Filter out aggregated components."""
        # aggregation of corresponding model terms, e.g. sum
        aggregated_components = {
            "additive_terms",
            "multiplicative_terms",
            "extra_regressors_additive",
            "extra_regressors_multiplicative",
        }

        return set(components) - aggregated_components

    def _check_mul_components(self):
        """Raise error if model contains multiplicative components."""
        components_modes = self.model.component_modes
        if components_modes is None:
            raise ValueError("This model is not fitted!")

        mul_components = self._filter_aggregated_components(self.model.component_modes["multiplicative"])
        if len(mul_components) > 0:
            raise ValueError("Forecast decomposition is only supported for additive components!")

    def _predict_seasonal_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate seasonal, holidays and exogenous components."""
        model = self.model

        seasonal_features, _, component_cols, _ = model.make_all_seasonality_features(df)

        holiday_names = set(model.train_holiday_names) if model.train_holiday_names is not None else set()

        components_names = list(
            filter(lambda v: v not in holiday_names, self._filter_aggregated_components(component_cols.columns))
        )

        beta_c = model.params["beta"].T * component_cols[components_names].values
        comp = seasonal_features.values @ beta_c

        # apply rescaling for additive components
        comp *= model.y_scale

        return pd.DataFrame(data=comp, columns=components_names)

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
        self._check_mul_components()

        prophet_df = self._prepare_prophet_df(df=df)

        prophet_df = self.model.setup_dataframe(prophet_df)

        components = self._predict_seasonal_components(df=prophet_df)
        components["trend"] = self.model.predict_trend(df=prophet_df)

        return components.add_prefix("target_component_")

    def get_model(self) -> Prophet:
        """Get internal prophet.Prophet model that is used inside etna class.

        Returns
        -------
        result:
           Internal model
        """
        return self.model

    def __getstate__(self):
        state = self.__dict__.copy()
        try:
            model_dict = model_to_dict(self.model)
            is_fitted = True
        except ValueError:
            is_fitted = False
            model_dict = {}
        del state["model"]
        state["_is_fitted"] = is_fitted
        state["_model_dict"] = model_dict
        return state

    def __setstate__(self, state):
        local_state = deepcopy(state)
        is_fitted = local_state["_is_fitted"]
        model_dict = local_state["_model_dict"]
        del local_state["_is_fitted"]
        del local_state["_model_dict"]

        self.__dict__.update(local_state)

        if is_fitted:
            self.model = model_from_dict(model_dict)
        else:
            self.model = self._create_model()


class ProphetModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """Class for holding Prophet model.

    Notes
    -----
    Original Prophet can use features 'cap' and 'floor',
    they should be added to the known_future list on dataset initialization.

    This model supports in-sample and out-of-sample forecast decomposition. The number
    of components in the decomposition depends on model parameters. Main components are:
    trend, seasonality, holiday and exogenous effects. Seasonal components will be decomposed
    down to individual periods if fitted. Holiday and exogenous will be present in decomposition
    if fitted.Corresponding components are obtained directly from the model.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import ProphetModel
    >>> classic_df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> df = TSDataset.to_dataset(df=classic_df)
    >>> ts = TSDataset(df, freq="D")
    >>> future = ts.make_future(7)
    >>> model = ProphetModel(growth="flat")
    >>> model.fit(ts=ts)
    ProphetModel(growth = 'flat', changepoints = None, n_changepoints = 25,
    changepoint_range = 0.8, yearly_seasonality = 'auto', weekly_seasonality = 'auto',
    daily_seasonality = 'auto', holidays = None, seasonality_mode = 'additive',
    seasonality_prior_scale = 10.0, holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05,
    mcmc_samples = 0, interval_width = 0.8, uncertainty_samples = 1000, stan_backend = None,
    additional_seasonality_params = (), )
    >>> forecast = model.forecast(future)
    >>> forecast
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12      0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
    """

    def __init__(
        self,
        growth: str = "linear",
        changepoints: Optional[List[datetime]] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Union[str, bool] = "auto",
        weekly_seasonality: Union[str, bool] = "auto",
        daily_seasonality: Union[str, bool] = "auto",
        holidays: Optional[pd.DataFrame] = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.8,
        uncertainty_samples: Union[int, bool] = 1000,
        stan_backend: Optional[str] = None,
        additional_seasonality_params: Iterable[Dict[str, Union[str, float, int]]] = (),
    ):
        """
        Create instance of Prophet model.

        Parameters
        ----------
        growth:
            Options are ‘linear’ and ‘logistic’. This likely will not be tuned;
            if there is a known saturating point and growth towards that point
            it will be included and the logistic trend will be used, otherwise
            it will be linear.
        changepoints:
            List of dates at which to include potential changepoints. If
            not specified, potential changepoints are selected automatically.
        n_changepoints:
            Number of potential changepoints to include. Not used
            if input ``changepoints`` is supplied. If ``changepoints`` is not supplied,
            then ``n_changepoints`` potential changepoints are selected uniformly from
            the first ``changepoint_range`` proportion of the history.
        changepoint_range:
            Proportion of history in which trend changepoints will
            be estimated. Defaults to 0.8 for the first 80%. Not used if
            ``changepoints`` is specified.
        yearly_seasonality:
            By default (‘auto’) this will turn yearly seasonality on if there is
            a year of data, and off otherwise. Options are [‘auto’, True, False].
            If there is more than a year of data, rather than trying to turn this
            off during HPO, it will likely be more effective to leave it on and
            turn down seasonal effects by tuning ``seasonality_prior_scale``.
        weekly_seasonality:
            Same as for ``yearly_seasonality``.
        daily_seasonality:
            Same as for ``yearly_seasonality``.
        holidays:
            ``pd.DataFrame`` with columns holiday (string) and ds (date type)
            and optionally columns lower_window and upper_window which specify a
            range of days around the date to be included as holidays.
            ``lower_window=-2`` will include 2 days prior to the date as holidays. Also
            optionally can have a column ``prior_scale`` specifying the prior scale for
            that holiday.
        seasonality_mode:
            'additive' (default) or 'multiplicative'.
        seasonality_prior_scale:
            Parameter modulating the strength of the
            seasonality model. Larger values allow the model to fit larger seasonal
            fluctuations, smaller values dampen the seasonality. Can be specified
            for individual seasonalities using ``add_seasonality``.
        holidays_prior_scale:
            Parameter modulating the strength of the holiday components model, unless overridden
            in the holidays input.
        changepoint_prior_scale:
            Parameter modulating the flexibility of the
            automatic changepoint selection. Large values will allow many
            changepoints, small values will allow few changepoints.
        mcmc_samples:
            Integer, if greater than 0, will do full Bayesian inference
            with the specified number of MCMC samples. If 0, will do MAP
            estimation.
        interval_width:
            Float, width of the uncertainty intervals provided
            for the forecast. If ``mcmc_samples=0``, this will be only the uncertainty
            in the trend using the MAP estimate of the extrapolated generative
            model. If ``mcmc.samples>0``, this will be integrated over all model
            parameters, which will include uncertainty in seasonality.
        uncertainty_samples:
            Number of simulated draws used to estimate
            uncertainty intervals. Settings this value to 0 or False will disable
            uncertainty estimation and speed up the calculation.
        stan_backend:
            as defined in StanBackendEnum default: None - will try to
            iterate over all available backends and find the working one
        additional_seasonality_params: Iterable[Dict[str, Union[int, float, str]]]
            parameters that describe additional (not 'daily', 'weekly', 'yearly') seasonality that should be
            added to model; dict with required keys 'name', 'period', 'fourier_order' and optional ones 'prior_scale',
            'mode', 'condition_name' will be used for :py:meth:`prophet.Prophet.add_seasonality` method call.
        """
        self.growth = growth
        self.n_changepoints = n_changepoints
        self.changepoints = changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        self.additional_seasonality_params = additional_seasonality_params

        super(ProphetModel, self).__init__(
            base_model=_ProphetAdapter(
                growth=self.growth,
                n_changepoints=self.n_changepoints,
                changepoints=self.changepoints,
                changepoint_range=self.changepoint_range,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=self.holidays,
                seasonality_mode=self.seasonality_mode,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                changepoint_prior_scale=self.changepoint_prior_scale,
                mcmc_samples=self.mcmc_samples,
                interval_width=self.interval_width,
                uncertainty_samples=self.uncertainty_samples,
                stan_backend=self.stan_backend,
                additional_seasonality_params=self.additional_seasonality_params,
            )
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``seasonality_mode``, ``seasonality_prior_scale``, ``changepoint_prior_scale``,
        ``changepoint_range``, ``holidays_prior_scale``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "seasonality_mode": CategoricalDistribution(["additive", "multiplicative"]),
            "seasonality_prior_scale": FloatDistribution(low=1e-2, high=10, log=True),
            "changepoint_prior_scale": FloatDistribution(low=1e-3, high=0.5, log=True),
            "changepoint_range": FloatDistribution(low=0.8, high=0.95),
            "holidays_prior_scale": FloatDistribution(low=1e-2, high=10, log=True),
        }
