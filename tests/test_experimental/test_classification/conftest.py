import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


@pytest.fixture()
def x_y():
    x = np.random.random((5, 7))
    y = np.array([1, 0, 0, 1, 0])
    return x, y


@pytest.fixture()
def many_time_series():
    x = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0])]
    y = np.array([1, 0, 1])
    return x, y


@pytest.fixture
def many_time_series_ts(many_time_series):
    x, y = many_time_series
    dfs = []
    ts_y = {}
    for i, series in enumerate(x):
        df = generate_ar_df(periods=10, n_segments=1, start_time="2000-01-01")
        df = df.iloc[-len(series) :]
        df["target"] = series
        df["segment"] = f"segment_{i}"
        ts_y[f"segment_{i}"] = y[i]
        dfs.append(df)
    df = pd.concat(dfs)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts, ts_y
