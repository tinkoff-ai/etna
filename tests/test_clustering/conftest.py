from typing import Tuple

import pandas as pd
import pytest

from etna.datasets import TSDataset


@pytest.fixture
def two_series() -> Tuple[pd.Series, pd.Series]:
    """Generate two series with different timestamp range."""
    x1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=10)})
    x1["target"] = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    x1.set_index("timestamp", inplace=True)

    x2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-02", periods=10)})
    x2["target"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x2.set_index("timestamp", inplace=True)

    return x1["target"], x2["target"]


@pytest.fixture
def pattern():
    x = [1] * 5 + [20, 3, 1, -5, -7, -8, -9, -10, -7.5, -6.5, -5, -4, -3, -2, -1, 0, 0, 1, 1] + [-1] * 11
    return x


@pytest.fixture
def dtw_ts(pattern) -> TSDataset:
    """Get df with complex pattern with timestamp lag."""
    df = pd.DataFrame()
    for i in range(1, 8):
        date_range = pd.date_range(f"2020-01-0{str(i)}", periods=35)
        tmp = pd.DataFrame({"timestamp": date_range})
        tmp["segment"] = str(i)
        tmp["target"] = pattern
        df = df.append(tmp, ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D")
    return ts
