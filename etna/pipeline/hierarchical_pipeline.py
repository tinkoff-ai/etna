import pathlib
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import pandas as pd

from etna.datasets.tsdataset import TSDataset
from etna.datasets.utils import get_target_with_quantiles
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import Metric
from etna.models.base import ModelType
from etna.pipeline.pipeline import Pipeline
from etna.reconciliation.base import BaseReconciliator
from etna.transforms.base import Transform


class HierarchicalPipeline(Pipeline):
    """Pipeline of transforms with a final estimator for hierarchical time series data.

    Notes
    -----
    Aggregation of target quantiles and components is performed along
    with the target itself. It uses a provided hierarchical structure and a reconciliation method.
    """

    def __init__(
        self, reconciliator: BaseReconciliator, model: ModelType, transforms: Sequence[Transform] = (), horizon: int = 1
    ):
        """Create instance of HierarchicalPipeline with given parameters.

        Parameters
        ----------
        reconciliator:
            Instance of reconciliation method
        model:
            Instance of the etna Model
        transforms:
            Sequence of the transforms
        horizon:
             Number of timestamps in the future for forecasting

        Warnings
        --------
        Estimation of forecast intervals with `forecast(prediction_interval=True)` method and
        `BottomUpReconciliator` may be not reliable.
        """
        super().__init__(model=model, transforms=transforms, horizon=horizon)
        self.reconciliator = reconciliator
        self._fit_ts: Optional[TSDataset] = None

    def fit(self, ts: TSDataset) -> "HierarchicalPipeline":
        """Fit the HierarchicalPipeline.

        Fit and apply given transforms to the data, then fit the model on the transformed data.
        Provided hierarchical dataset will be aggregated to the source level before fitting pipeline.

        Parameters
        ----------
        ts:
            Dataset with hierarchical timeseries data

        Returns
        -------
        :
            Fitted HierarchicalPipeline instance
        """
        self._fit_ts = deepcopy(ts)

        self.reconciliator.fit(ts=ts)
        ts = self.reconciliator.aggregate(ts=ts)
        super().fit(ts=ts)
        return self

    def raw_forecast(
        self,
        ts: TSDataset,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.25, 0.75),
        n_folds: int = 3,
        return_components: bool = False,
    ) -> TSDataset:
        """Make a forecast of the next points of a dataset at the source level.

        The result of forecasting starts from the last point of ``ts``, not including it.

        Parameters
        ----------
        ts:
            Dataset to forecast
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions at the source level
        """
        # handle `prediction_interval=True` separately
        source_ts = self.reconciliator.aggregate(ts=ts)
        forecast = super().forecast(
            ts=source_ts, prediction_interval=False, n_folds=n_folds, return_components=return_components
        )
        if prediction_interval:
            forecast = self._forecast_prediction_interval(
                ts=ts, predictions=forecast, quantiles=quantiles, n_folds=n_folds
            )

        target_columns = tuple(get_target_with_quantiles(columns=forecast.columns))
        hierarchical_forecast = TSDataset(
            df=forecast[..., target_columns],
            freq=forecast.freq,
            df_exog=forecast.df_exog,
            known_future=forecast.known_future,
            hierarchical_structure=ts.hierarchical_structure,  # type: ignore
        )

        if return_components:
            hierarchical_forecast.add_target_components(target_components_df=forecast.get_target_components())

        return hierarchical_forecast

    def raw_predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
    ) -> TSDataset:
        """Make in-sample predictions on dataset at the source level in a given range.

        Parameters
        ----------
        ts:
            Dataset to make predictions on. If not given, dataset given during :py:meth:``fit`` is used.
        start_timestamp:
            First timestamp of prediction range to return, should be >= than first timestamp in ``ts``;
            expected that beginning of each segment <= ``start_timestamp``;
            if isn't set the first timestamp where each segment began is taken.
        end_timestamp:
            Last timestamp of prediction range to return; if isn't set the last timestamp of ``ts`` is taken.
            Expected that value is less or equal to the last timestamp in ``ts``.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components.

        Returns
        -------
        :
            Dataset with predictions at the source level in ``[start_timestamp, end_timestamp]`` range.
        """
        source_ts = self.reconciliator.aggregate(ts=ts)
        forecast = super().predict(
            ts=source_ts,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            return_components=return_components,
        )

        target_columns = tuple(get_target_with_quantiles(columns=forecast.columns))
        hierarchical_forecast = TSDataset(
            df=forecast[..., target_columns],
            freq=forecast.freq,
            df_exog=forecast.df_exog,
            known_future=forecast.known_future,
            hierarchical_structure=ts.hierarchical_structure,  # type: ignore
        )

        if return_components:
            hierarchical_forecast.add_target_components(target_components_df=forecast.get_target_components())

        return hierarchical_forecast

    def forecast(
        self,
        ts: Optional[TSDataset] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        n_folds: int = 3,
        return_components: bool = False,
    ) -> TSDataset:
        """Make a forecast of the next points of a dataset at a target level.

        The result of forecasting starts from the last point of ``ts``, not including it.

        Method makes a prediction for target at the source level of hierarchy and then makes reconciliation to target level.

        Parameters
        ----------
        ts:
            Dataset to forecast. If not given, dataset given during :py:meth:``fit`` is used.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions at the target level of hierarchy.
        """
        if ts is None:
            if self._fit_ts is None:
                raise ValueError(
                    "There is no ts to forecast! Pass ts into forecast method or make sure that pipeline is loaded with ts."
                )
            ts = self._fit_ts

        forecast = self.raw_forecast(
            ts=ts,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            n_folds=n_folds,
            return_components=return_components,
        )
        forecast_reconciled = self.reconciliator.reconcile(forecast)
        return forecast_reconciled

    def predict(
        self,
        ts: Optional[TSDataset] = None,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
    ) -> TSDataset:
        """Make in-sample predictions on dataset at the target level in a given range.

        Method makes a prediction for target at the source level of hierarchy and then makes reconciliation to the target level.

        Currently, in situation when segments start with different timestamps
        we only guarantee to work with ``start_timestamp`` >= beginning of all segments.

        Parameters
        ----------
        ts:
            Dataset to make predictions on. If not given, dataset given during :py:meth:``fit`` is used.
        start_timestamp:
            First timestamp of prediction range to return, should be >= than first timestamp in ``ts``;
            expected that beginning of each segment <= ``start_timestamp``;
            if isn't set the first timestamp where each segment began is taken.
        end_timestamp:
            Last timestamp of prediction range to return; if isn't set the last timestamp of ``ts`` is taken.
            Expected that value is less or equal to the last timestamp in ``ts``.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components.

        Returns
        -------
        :
            Dataset with predictions at the target level in ``[start_timestamp, end_timestamp]`` range.
        """
        if ts is None:
            if self._fit_ts is None:
                raise ValueError(
                    "There is no ts to predict! Pass ts into predict method or make sure that pipeline is loaded with ts."
                )
            ts = self._fit_ts

        forecast = self.raw_predict(
            ts=ts,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            return_components=return_components,
        )
        forecast_reconciled = self.reconciliator.reconcile(forecast)
        return forecast_reconciled

    def _compute_metrics(
        self, metrics: List[Metric], y_true: TSDataset, y_pred: TSDataset
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for given y_true, y_pred."""
        if y_true.current_df_level != self.reconciliator.target_level:
            y_true = y_true.get_level_dataset(self.reconciliator.target_level)

        if y_pred.current_df_level == self.reconciliator.source_level:
            y_pred = self.reconciliator.reconcile(y_pred)

        metrics_values: Dict[str, Dict[str, float]] = {}
        for metric in metrics:
            metrics_values[metric.name] = metric(y_true=y_true, y_pred=y_pred)  # type: ignore
        return metrics_values

    def _forecast_prediction_interval(
        self, ts: TSDataset, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Add prediction intervals to the forecasts."""
        if self.ts is None:
            raise ValueError("Pipeline is not fitted! Fit the Pipeline before calling forecast method.")

        self.forecast, self.raw_forecast = self.raw_forecast, self.forecast  # type: ignore
        try:
            # TODO: rework intervals estimation for `BottomUpReconciliator`

            with tslogger.disable():
                _, forecasts, _ = self.backtest(ts=ts, metrics=[MAE()], n_folds=n_folds)

            source_ts = self.reconciliator.aggregate(ts=ts)
            self._add_forecast_borders(
                ts=source_ts, backtest_forecasts=forecasts, quantiles=quantiles, predictions=predictions
            )
            return predictions

        finally:
            self.forecast, self.raw_forecast = self.raw_forecast, self.forecast  # type: ignore

    def save(self, path: pathlib.Path):
        """Save the object.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        fit_ts = self._fit_ts

        try:
            # extract attributes we can't easily save
            delattr(self, "_fit_ts")

            # save the remaining part
            super().save(path=path)
        finally:
            self._fit_ts = fit_ts

    @classmethod
    def load(cls, path: pathlib.Path, ts: Optional[TSDataset] = None) -> "HierarchicalPipeline":
        """Load an object.

        Parameters
        ----------
        path:
            Path to load object from.
        ts:
            TSDataset to set into loaded pipeline.

        Returns
        -------
        :
            Loaded object.
        """
        obj = super().load(path=path)
        obj._fit_ts = deepcopy(ts)
        if ts is not None:
            obj.ts = obj.reconciliator.aggregate(ts=ts)
        else:
            obj.ts = None
        return obj
