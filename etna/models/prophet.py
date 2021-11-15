from datetime import datetime
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd

from etna import SETTINGS
from etna.datasets import TSDataset
from etna.models.base import PerSegmentModel
from etna.models.base import log_decorator

if SETTINGS.prophet_required:
    from prophet import Prophet


class _ProphetModel:
    """Class for holding Prophet model."""

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
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        self.additional_seasonality_params = additional_seasonality_params

        self.model = Prophet(
            growth=self.growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend,
        )

        for seasonality_params in self.additional_seasonality_params:
            self.model.add_seasonality(**seasonality_params)

    def fit(self, df: pd.DataFrame) -> "_ProphetModel":
        """
        Fits a Prophet model.

        Parameters
        ----------
        df:
            Features dataframe
        """
        prophet_df = pd.DataFrame()
        prophet_df["y"] = df["target"]
        prophet_df["ds"] = df["timestamp"]
        for column_name in df.columns:
            if column_name.startswith("regressor"):
                if column_name in ["regressor_cap", "regressor_floor"]:
                    prophet_column_name = column_name[len("regressor_") :]
                else:
                    self.model.add_regressor(column_name)
                    prophet_column_name = column_name
                prophet_df[prophet_column_name] = df[column_name]
        self.model.fit(prophet_df)
        return self

    def predict(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]):
        """
        Compute Prophet predictions.

        Parameters
        ----------
        df :
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
        df = df.reset_index()
        prophet_df = pd.DataFrame()
        prophet_df["y"] = df["target"]
        prophet_df["ds"] = df["timestamp"]
        for column_name in df.columns:
            if column_name.startswith("regressor"):
                if column_name in ["regressor_cap", "regressor_floor"]:
                    prophet_column_name = column_name[len("regressor_") :]
                else:
                    prophet_column_name = column_name
                prophet_df[prophet_column_name] = df[column_name]

        forecast = self.model.predict(prophet_df)
        y_pred = pd.DataFrame(forecast["yhat"])
        if prediction_interval:
            sim_values = self.model.predictive_samples(prophet_df)
            for quantile in quantiles:
                percentile = quantile * 100
                y_pred[f"yhat_{quantile:.4g}"] = self.model.percentile(sim_values["yhat"], percentile, axis=1)
        return y_pred


class ProphetModel(PerSegmentModel):
    """Class for holding Prophet model.

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
    seasonality_prior_scale = 10.0, holidays_prior_scale = 10.0, mcmc_samples = 0,
    interval_width = 0.8, uncertainty_samples = 1000, stan_backend = None,
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
            if input `changepoints` is supplied. If `changepoints` is not supplied,
            then n_changepoints potential changepoints are selected uniformly from
            the first `changepoint_range` proportion of the history.
        changepoint_range:
            Proportion of history in which trend changepoints will
            be estimated. Defaults to 0.8 for the first 80%. Not used if
            `changepoints` is specified.
        yearly_seasonality:
            By default (‘auto’) this will turn yearly seasonality on if there is
            a year of data, and off otherwise. Options are [‘auto’, True, False].
            If there is more than a year of data, rather than trying to turn this
            off during HPO, it will likely be more effective to leave it on and
            turn down seasonal effects by tuning seasonality_prior_scale.
        weekly_seasonality:
            Same as for yearly_seasonality.
        daily_seasonality:
            Same as for yearly_seasonality.
        holidays:
            pd.DataFrame with columns holiday (string) and ds (date type)
            and optionally columns lower_window and upper_window which specify a
            range of days around the date to be included as holidays.
            lower_window=-2 will include 2 days prior to the date as holidays. Also
            optionally can have a column prior_scale specifying the prior scale for
            that holiday.
        seasonality_mode:
            'additive' (default) or 'multiplicative'.
        seasonality_prior_scale:
            Parameter modulating the strength of the
            seasonality model. Larger values allow the model to fit larger seasonal
            fluctuations, smaller values dampen the seasonality. Can be specified
            for individual seasonalities using add_seasonality.
        holidays_prior_scale:
            Parameter modulating the strength of the holiday components model, unless overridden
            in the holidays input.
        mcmc_samples:
            Integer, if greater than 0, will do full Bayesian inference
            with the specified number of MCMC samples. If 0, will do MAP
            estimation.
        interval_width:
            Float, width of the uncertainty intervals provided
            for the forecast. If mcmc_samples=0, this will be only the uncertainty
            in the trend using the MAP estimate of the extrapolated generative
            model. If mcmc.samples>0, this will be integrated over all model
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
            'mode', 'condition_name' will be used for prophet.Prophet().add_seasonality method call.

        Notes
        -----
        Original Prophet can use features 'cap' and 'floor',
        but our wrapper expects it under names 'regressor_cap' and 'regressor_floor'.
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
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        self.additional_seasonality_params = additional_seasonality_params

        super(ProphetModel, self).__init__(
            base_model=_ProphetModel(
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
                mcmc_samples=self.mcmc_samples,
                interval_width=self.interval_width,
                uncertainty_samples=self.uncertainty_samples,
                stan_backend=self.stan_backend,
                additional_seasonality_params=self.additional_seasonality_params,
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
            column: column.replace("yhat", "target") for column in segment_predict.columns if column.startswith("yhat")
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
        TSDataset
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
