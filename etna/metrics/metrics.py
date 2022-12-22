from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import sign
from etna.metrics import smape
from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode

import numpy as np


class MAE(Metric):
    """Mean absolute error metric with multi-segment computation support.

    .. math::
        MAE(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}{\\mid y\_true_i - y\_pred_i \\mid}}{n}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mae, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MSE(Metric):
    """Mean squared error metric with multi-segment computation support.

    .. math::
        MSE(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}{(y\_true_i - y\_pred_i)^2}}{n}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mse, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False

class RMSE(Metric):
    """Root mean squared error metric with multi-segment computation support.

    .. math::
        RMSE(y\_true, y\_pred) = \\sqrt{\\frac{\\sum_{i=0}^{n-1}{(y\_true_i - y\_pred_i)^2}}{n}}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, squared: bool = False,  **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mse, **kwargs)
        self.squared = squared

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
        """
        Compute metric's value with ``y_true`` and ``y_pred``.

        Notes
        -----
        Note that if ``y_true`` and ``y_pred`` are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
        :
            metric's value aggregated over segments or not (depends on mode)
        """
        self._log_start()
        self._validate_segment_columns(y_true=y_true, y_pred=y_pred)

        segments = set(y_true.df.columns.get_level_values("segment"))
        metrics_per_segment = {}
        for segment in segments:
            self._validate_timestamp_columns(
                timestamp_true=y_true[:, segment, "target"].dropna().index,
                timestamp_pred=y_pred[:, segment, "target"].dropna().index,
            )
            if self.squared:
                metrics_per_segment[segment] = np.sqrt(
                    self.metric_fn(
                        y_true=y_true[:, segment, "target"].values, y_pred=y_pred[:, segment, "target"].values, **self.kwargs
                    )
                )
            else:
                metrics_per_segment[segment] = self.metric_fn(
                        y_true=y_true[:, segment, "target"].values, y_pred=y_pred[:, segment, "target"].values,
                        **self.kwargs
                    )
        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False

class R2(Metric):
    """Coefficient of determination metric with multi-segment computation support.

    .. math::
        R^2(y\_true, y\_pred) = 1 - \\frac{\\sum_{i=0}^{n-1}{(y\_true_i - y\_pred_i)^2}}{\\sum_{i=0}^{n-1}{(y\_true_i - \\overline{y\_true})^2}}
    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=r2_score, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return True


class MAPE(Metric):
    """Mean absolute percentage error metric with multi-segment computation support.

    .. math::
       MAPE(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\frac{\\sum_{i=0}^{n-1}{\\mid y\_true_i - y\_pred_i\\mid}}{\\mid y\_true_i \\mid + \epsilon}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mape, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class SMAPE(Metric):
    """Symmetric mean absolute percentage error metric with multi-segment computation support.

    .. math::
       SMAPE(y\_true, y\_pred) = \\frac{2 \\cdot 100 \\%}{n}\\cdot\\frac{\\sum_{i=0}^{n-1}{\\mid y\_true_i - y\_pred_i\\mid}}{\\mid y\_true_i \\mid + \\mid y\_pred_i \\mid}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=smape, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MedAE(Metric):
    """Median absolute error metric with multi-segment computation support.

    .. math::
       MedAE(y\_true, y\_pred) = median(\\mid y\_true_1 - y\_pred_1 \\mid, \\cdots, \\mid y\_true_n - y\_pred_n \\mid)

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=medae, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class MSLE(Metric):
    """Mean squared logarithmic error metric with multi-segment computation support.

    .. math::
       MSLE(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=0}^{n - 1}{(ln(1 + y\_true_i) - ln(1 + y\_pred_i))^2}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments

        """
        super().__init__(mode=mode, metric_fn=msle, **kwargs)

    @property
    def greater_is_better(self) -> bool:
        """Whether higher metric value is better."""
        return False


class Sign(Metric):
    """Sign error metric with multi-segment computation support.

    .. math::
        Sign(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=0}^{n - 1}{sign(y\_true_i - y\_pred_i)}

    Notes
    -----
    You can read more about logic of multi-segment metrics in Metric docs.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """Init metric.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=sign, **kwargs)

    @property
    def greater_is_better(self) -> None:
        """Whether higher metric value is better."""
        return None


__all__ = ["MAE", "MSE", "RMSE", "R2", "MSLE", "MAPE", "SMAPE", "MedAE", "Sign"]
