import functools

import numpy as np
import pytest

from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode


def to_be_fixed(raises, match=None):
    def to_be_fixed_concrete(func):
        @functools.wraps(func)
        def wrapped_test(*args, **kwargs):
            with pytest.raises(raises, match=match):
                return func(*args, **kwargs)

        return wrapped_test

    return to_be_fixed_concrete


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

    @property
    def greater_is_better(self) -> bool:
        return False
