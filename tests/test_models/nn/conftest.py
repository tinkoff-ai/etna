from typing import Tuple

import pandas as pd
import pytest

from etna.datasets import TSDataset


@pytest.fixture()
def weekly_period_df(n_repeats=15):
    segment_1 = [7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 1.0]
    segment_2 = [7.0, 7.0, 7.0, 4.0, 1.0, 7.0, 7.0]
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=n_repeats * len(segment_1)))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 * n_repeats + segment_2 * n_repeats,
            "segment": ["segment_1"] * n_repeats * len(segment_1) + ["segment_2"] * n_repeats * len(segment_2),
        }
    )
    return df


@pytest.fixture()
def ts_dataset_weekly_function_with_horizon(weekly_period_df):
    def wrapper(horizon: int) -> Tuple[TSDataset, TSDataset]:
        ts_start = sorted(set(weekly_period_df.timestamp))[-horizon]
        train, test = (
            weekly_period_df[lambda x: x.timestamp < ts_start],
            weekly_period_df[lambda x: x.timestamp >= ts_start],
        )

        ts_train = TSDataset(TSDataset.to_dataset(train), "D")
        ts_test = TSDataset(TSDataset.to_dataset(test), "D")
        return ts_train, ts_test

    return wrapper
