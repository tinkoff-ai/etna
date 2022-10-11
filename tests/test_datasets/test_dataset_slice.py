from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset

@pytest.fixture
def ts() -> TSDataset:
    periods = 10
    df1 = pd.DataFrame({"timestamp": pd.date_range("2000-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = 1

    df2 = pd.DataFrame({"timestamp": pd.date_range("2000-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = 10

    exog_weekend_1 = pd.DataFrame({"timestamp": pd.date_range("2000-01-01", periods=periods + 7)})
    exog_weekend_1["segment"] = "segment_1"
    exog_weekend_1["exog"] = 100

    exog_weekend_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods + 7)})
    exog_weekend_2["segment"] = "segment_2"
    exog_weekend_2["exog"] = 1000

    df = pd.concat([df1, df2]).reset_index(drop=True)
    exog = pd.concat([exog_weekend_1, exog_weekend_2]).reset_index(drop=True)

    df = TSDataset.to_dataset(df)
    exog = TSDataset.to_dataset(exog)

    tsds = TSDataset(df, freq="D", df_exog=exog, known_future="all")

    return tsds


@pytest.mark.parametrize("idx, expected_dataset", [(0), ("2000-01-01"), (slice(None)), (slice(start="2000-01-01", stop="2000-01-02"))])
def test_time_index_slice(ts, idx, expected_dataset):
    dataset_slice = ts[idx]
    assert dataset_slice == expected_dataset

@pytest.mark.parametrize("idx, expected_dataset", [(slice(start="2000-01-01", stop="2000-01-02"), "target")])
def test_time_feature_index_slice(ts, idx, expected_dataset):
    dataset_slice = ts[idx]
    assert dataset_slice == expected_dataset

@pytest.mark.parametrize("idx, expected_dataset", [(slice(start="2000-01-01", stop="2000-01-02"), "segment_1", "target")])
def test_time_segment_feature_index_slice(ts, idx, expected_dataset):
    dataset_slice = ts[idx]
    assert dataset_slice == expected_dataset
