from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

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
    """Pipeline of transforms with a final estimator for hierarchical time series data."""

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
        self, prediction_interval: bool = False, quantiles: Sequence[float] = (0.25, 0.75), n_folds: int = 3
    ) -> TSDataset:
        """Make a prediction for target at the source level of hierarchy.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation

        Returns
        -------
        :
            Dataset with predictions at the source level
        """
        forecast = super().forecast(prediction_interval=prediction_interval, quantiles=quantiles, n_folds=n_folds)
        target_columns = tuple(get_target_with_quantiles(columns=forecast.columns))

        hierarchical_forecast = TSDataset(
            df=forecast[..., target_columns],
            freq=forecast.freq,
            df_exog=forecast.df_exog,
            known_future=forecast.known_future,
            hierarchical_structure=self.ts.hierarchical_structure,  # type: ignore
        )
        return hierarchical_forecast

    def forecast(
        self, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975), n_folds: int = 3
    ) -> TSDataset:
        """Make a prediction for target at the source level of hierarchy and make reconciliation to target level.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation

        Returns
        -------
        :
            Dataset with predictions at the target level of hierarchy.
        """
        forecast = self.raw_forecast(prediction_interval=prediction_interval, quantiles=quantiles, n_folds=n_folds)
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
        self, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Add prediction intervals to the forecasts."""
        self.forecast, self.raw_forecast = self.raw_forecast, self.forecast  # type: ignore

        if self.ts is None or self._fit_ts is None:
            raise ValueError("Pipeline is not fitted! Fit the Pipeline before calling forecast method.")

        # TODO: rework intervals estimation for `BottomUpReconciliator`

        with tslogger.disable():
            _, forecasts, _ = self.backtest(ts=self._fit_ts, metrics=[MAE()], n_folds=n_folds)

        self._add_forecast_borders(backtest_forecasts=forecasts, quantiles=quantiles, predictions=predictions)

        self.forecast, self.raw_forecast = self.raw_forecast, self.forecast  # type: ignore

        return predictions
