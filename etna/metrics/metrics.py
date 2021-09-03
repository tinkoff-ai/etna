from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import smape
from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode


class MAE(Metric):
    """Mean absolute error metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Mean absolute error metric with multi-segment computation support.

        .. math::
            MAE(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}{\\mid y\_true_i - y\_pred_i \\mid}}{n}

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mae, **kwargs)


class MSE(Metric):
    """Mean squared error metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Create instance of Mean squared error metric with multi-segment computation support.

        .. math::
            MSE(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n-1}{(y\_true_i - y\_pred_i)^2}}{n}

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mse, **kwargs)


class R2(Metric):
    """Coefficient of determination metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Create instance of Coefficient of determination metric with multi-segment computation support.

        .. math::
            R^2(y\_true, y\_pred) = 1 - \\frac{\\sum_{i=0}^{n-1}{(y\_true_i - y\_pred_i)^2}}{\\sum_{i=0}^{n-1}{(y\_true_i - \\overline{y\_true})^2}}

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=r2_score, **kwargs)


class MAPE(Metric):
    """Mean absolute percentage error metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Mean absolute percentage error metric with multi-segment computation support.

        .. math::
           MAPE(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\frac{\\sum_{i=0}^{n-1}{\\mid y\_true_i - y\_pred_i\\mid}}{\\mid y\_true_i \\mid + \epsilon}

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=mape, **kwargs)


class SMAPE(Metric):
    """Symmetric mean absolute percentage error metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Symmetric mean absolute percentage error metric with multi-segment computation support.

        .. math::
           SMAPE(y\_true, y\_pred) = \\frac{2 \\cdot 100 \\%}{n}\\cdot\\frac{\\sum_{i=0}^{n-1}{\\mid y\_true_i - y\_pred_i\\mid}}{\\mid y\_true_i \\mid + \\mid y\_pred_i \\mid}

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=smape, **kwargs)


class MedAE(Metric):
    """Median absolute error metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Create instance of Median absolute error metric with multi-segment computation support.

        .. math::
           MedAE(y\_true, y\_pred) = median(\\mid y\_true_1 - y\_pred_1 \\mid, \\cdots, \\mid y\_true_n - y\_pred_n \\mid)

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=medae, **kwargs)


class MSLE(Metric):
    """Mean squared logarithmic error metric."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Mean squared logarithmic error metric with multi-segment computation support.

        .. math::
           MSLE(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=0}^{n - 1}{(ln(1 + y\_true_i) - ln(1 + y\_pred_i))^2}

        Notes:
        You can read more about logic of multi-segment metrics in Metric docs.

        Parameters
        ----------
        mode: 'macro' or 'per-segment'
            metrics aggregation mode
        kwargs:
            metric's computation arguments
        """
        super().__init__(mode=mode, metric_fn=msle, **kwargs)


__all__ = ["MAE", "MSE", "R2", "MSLE", "MAPE", "SMAPE", "MedAE"]
