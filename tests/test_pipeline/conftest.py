from typing import Tuple

import pandas as pd
import pytest
import scipy
from numpy.random import RandomState
from scipy.stats import norm

from etna.datasets import TSDataset
from etna.models import CatBoostModelPerSegment
from etna.pipeline import Pipeline
from etna.transforms import LagTransform

INTERVAL_WIDTH = 0.95


@pytest.fixture
def catboost_pipeline() -> Pipeline:
    """Generate pipeline with CatBoostModelMultiSegment."""
    pipeline = Pipeline(
        model=CatBoostModelPerSegment(),
        transforms=[LagTransform(in_column="target", lags=[10, 11, 12], out_column="regressor_lag_feature")],
        horizon=7,
    )
    return pipeline


@pytest.fixture
def catboost_pipeline_big() -> Pipeline:
    """Generate pipeline with CatBoostModelMultiSegment."""
    pipeline = Pipeline(
        model=CatBoostModelPerSegment(),
        transforms=[LagTransform(in_column="target", lags=[25, 26, 27], out_column="regressor_lag_feature")],
        horizon=24,
    )
    return pipeline


@pytest.fixture
def weekly_period_ts(n_repeats: int = 15, horizon: int = 7) -> Tuple["TSDataset", "TSDataset"]:
    segment_1 = [7.0, 7.0, 3.0, 1.0]
    segment_2 = [40.0, 70.0, 20.0, 10.0]
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=n_repeats * len(segment_1)))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 * n_repeats + segment_2 * n_repeats,
            "segment": ["segment_1"] * n_repeats * len(segment_1) + ["segment_2"] * n_repeats * len(segment_2),
        }
    )
    ts_start = sorted(set(df.timestamp))[-horizon]
    train, test = (
        df[lambda x: x.timestamp < ts_start],
        df[lambda x: x.timestamp >= ts_start],
    )
    train = TSDataset(TSDataset.to_dataset(train), "D")
    test = TSDataset(TSDataset.to_dataset(test), "D")

    return train, test


@pytest.fixture
def splited_piecewise_constant_ts(
    first_constant_len=40, constant_1_1=7, constant_1_2=2, constant_2_1=50, constant_2_2=10, horizon=5
) -> Tuple["TSDataset", "TSDataset"]:

    segment_1 = [constant_1_1] * first_constant_len + [constant_1_2] * horizon * 2
    segment_2 = [constant_2_1] * first_constant_len + [constant_2_2] * horizon * 2

    quantile = norm.ppf(q=(1 + INTERVAL_WIDTH) / 2)
    se_1 = scipy.stats.sem([0] * horizon * 2 + [constant_1_1 - constant_1_2] * horizon)
    se_2 = scipy.stats.sem([0] * horizon * 2 + [constant_2_1 - constant_2_2] * horizon)
    lower = [x - se_1 * quantile for x in segment_1] + [x - se_2 * quantile for x in segment_2]
    upper = [x + se_1 * quantile for x in segment_1] + [x + se_2 * quantile for x in segment_2]

    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=len(segment_1)))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            "target_lower": lower,
            "target_upper": upper,
            "segment": ["segment_1"] * len(segment_1) + ["segment_2"] * len(segment_2),
        }
    )
    ts_start = sorted(set(df.timestamp))[-horizon]
    train, test = (
        df[lambda x: x.timestamp < ts_start],
        df[lambda x: x.timestamp >= ts_start],
    )
    train = TSDataset(TSDataset.to_dataset(train.drop(["target_lower", "target_upper"], axis=1)), "D")
    test = TSDataset(TSDataset.to_dataset(test), "D")
    return train, test


@pytest.fixture
def constant_ts(size=40) -> TSDataset:
    segment_1 = [7] * size
    segment_2 = [50] * size
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            "segment": ["segment_1"] * size + ["segment_2"] * size,
        }
    )
    ts = TSDataset(TSDataset.to_dataset(df), "D")
    return ts


@pytest.fixture
def constant_noisy_ts(size=40, use_noise=True) -> TSDataset:
    noise = RandomState(seed=42).normal(scale=3, size=size * 2)
    segment_1 = [7] * size
    segment_2 = [50] * size
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 + segment_2,
            "segment": ["segment_1"] * size + ["segment_2"] * size,
        }
    )
    if use_noise:
        df.loc[:, "target"] += noise
    ts = TSDataset(TSDataset.to_dataset(df), "D")
    return ts
