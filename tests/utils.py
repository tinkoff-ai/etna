import numpy as np
import pandas as pd

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


def equals_with_nans(first_df: pd.DataFrame, second_df: pd.DataFrame) -> bool:
    """Compare two dataframes with consideration NaN == NaN is true."""
    if first_df.shape != second_df.shape:
        return False
    compare_result = (first_df.isna() & second_df.isna()) | (first_df == second_df)
    return np.all(compare_result)
