import functools
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
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


def select_segments_subset(ts: TSDataset, segments: List[str]) -> TSDataset:
    df = ts.raw_df.loc[:, pd.IndexSlice[segments, :]].copy()
    df = df.loc[ts.df.index]
    df_exog = ts.df_exog
    if df_exog is not None:
        df_exog = df_exog.loc[:, pd.IndexSlice[segments, :]].copy()
    known_future = ts.known_future
    freq = ts.freq
    subset_ts = TSDataset(df=df, df_exog=df_exog, known_future=known_future, freq=freq)
    return subset_ts


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
