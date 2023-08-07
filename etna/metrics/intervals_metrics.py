from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from etna.datasets import TSDataset
from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode
from etna.metrics.functional_metrics import ArrayLike


def dummy(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    return np.nan


class _QuantileMetricMixin:
    def _validate_tsdataset_quantiles(self, ts: TSDataset, quantiles: Sequence[float]) -> None:
        """Check if quantiles presented in y_pred."""
        features = set(ts.df.columns.get_level_values("feature"))
        for quantile in quantiles:
            assert f"target_{quantile:.4g}" in features, f"Quantile {quantile} is not presented in tsdataset."


class Coverage(Metric, _QuantileMetricMixin):
    """Coverage metric for prediction intervals - precenteage of samples in the interval ``[lower quantile, upper quantile]``.

    .. math::
        Coverage(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}{[ y\_true_i \\ge y\_pred_i^{lower\_quantile}] * [y\_true_i \\le y\_pred_i^{upper\_quantile}] }}{n}

    Notes
    -----
    Works just if quantiles presented in y_pred
    """

    def __init__(
        self, quantiles: Tuple[float, float] = (0.025, 0.975), mode: str = MetricAggregationMode.per_segment, **kwargs
    ):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=dummy, **kwargs)
        self.quantiles = quantiles

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
        """
        Compute metric's value with y_true and y_pred.

        Notes
        -----
        Note that if y_true and y_pred are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
            metric's value aggregated over segments or not (depends on mode)
        """
        self._validate_segments(y_true=y_true, y_pred=y_pred)
        self._validate_target_columns(y_true=y_true, y_pred=y_pred)
        self._validate_index(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)
        self._validate_tsdataset_quantiles(ts=y_pred, quantiles=self.quantiles)

        df_true = y_true[:, :, "target"].sort_index(axis=1)
        df_pred_lower = y_pred[:, :, f"target_{self.quantiles[0]:.4g}"].sort_index(axis=1)
        df_pred_upper = y_pred[:, :, f"target_{self.quantiles[1]:.4g}"].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        upper_quantile_flag = df_true.values <= df_pred_upper.values
        lower_quantile_flag = df_true.values >= df_pred_lower.values
        values = np.mean(upper_quantile_flag * lower_quantile_flag, axis=0)
        metrics_per_segment = dict(zip(segments, values))

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> None:
        """Whether higher metric value is better."""
        return None


class Width(Metric, _QuantileMetricMixin):
    """Mean width of prediction intervals.

    .. math::
        Width(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}\\mid y\_pred_i^{upper\_quantile} - y\_pred_i^{lower\_quantile} \\mid}{n}

    Notes
    -----
    Works just if quantiles presented in y_pred
    """

    def __init__(
        self, quantiles: Tuple[float, float] = (0.025, 0.975), mode: str = MetricAggregationMode.per_segment, **kwargs
    ):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=dummy, **kwargs)
        self.quantiles = quantiles

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
        """
        Compute metric's value with y_true and y_pred.

        Notes
        -----
        Note that if y_true and y_pred are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
            metric's value aggregated over segments or not (depends on mode)
        """
        self._validate_segments(y_true=y_true, y_pred=y_pred)
        self._validate_target_columns(y_true=y_true, y_pred=y_pred)
        self._validate_index(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)
        self._validate_tsdataset_quantiles(ts=y_pred, quantiles=self.quantiles)

        df_true = y_true[:, :, "target"].sort_index(axis=1)
        df_pred_lower = y_pred[:, :, f"target_{self.quantiles[0]:.4g}"].sort_index(axis=1)
        df_pred_upper = y_pred[:, :, f"target_{self.quantiles[1]:.4g}"].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        values = np.mean(np.abs(df_pred_upper.values - df_pred_lower.values), axis=0)
        metrics_per_segment = dict(zip(segments, values))

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


__all__ = ["Coverage", "Width"]
