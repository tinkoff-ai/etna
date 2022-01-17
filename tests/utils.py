import numpy as np

from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode


def create_dummy_functional_metric(alpha: float = 1.0):
    def dummy_functional_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return alpha

    return dummy_functional_metric


class DummyMetric(Metric):
    """Dummy metric returning always given parameter.

    We change the name property here.
    """

    def __init__(self, mode: str = MetricAggregationMode.per_segment, alpha: float = 1.0, **kwargs):
        self.alpha = alpha
        super().__init__(mode=mode, metric_fn=create_dummy_functional_metric(alpha), **kwargs)

    @property
    def name(self) -> str:
        return self.__repr__()
