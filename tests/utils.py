import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode
from etna.transforms.utils import TruncateTransform


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


@pytest.fixture()
def range_ts():
    periods = 100
    df = pd.DataFrame(
        {
            "segment": 0,
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": np.arange(0, periods),
            "feature": np.arange(3, periods + 3),
        }
    )
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


@pytest.mark.parametrize(
    "segments_to_truncate, expected_nan_positions",
    [([], []), ([3, 6, 7], [0, 3, 4]), (np.arange(3, 103), np.arange(0, 100))],
)
def test_truncate_transform(segments_to_truncate, expected_nan_positions, range_ts):
    transform = TruncateTransform(in_column="target", mask_column="feature", segments_to_truncate=segments_to_truncate)
    range_ts.fit_transform([transform])
    assert range_ts.to_pandas()[("0", "target")][expected_nan_positions].isna().all()
    expected_not_nan_positions = set(range(range_ts.to_pandas().shape[0])).difference(set(expected_nan_positions))
    assert not range_ts.to_pandas()[("0", "target")][expected_not_nan_positions].isna().any()
