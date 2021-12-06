from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset


@pytest.fixture(autouse=True)
def random_seed():
    """Fixture to fix random state for every test case."""
    import random

    import torch

    SEED = 121  # noqa: N806
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


@pytest.fixture()
def example_df(random_seed):
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def two_dfs_with_different_timestamps(random_seed):
    """Generate two dataframes with the same segments and different timestamps"""

    def generate_df(start_time):
        df = pd.DataFrame()
        for i in range(5):
            tmp = pd.DataFrame({"timestamp": pd.date_range(start_time, "2021-01-01")})
            tmp["segment"] = f"segment_{i + 1}"
            tmp["target"] = np.random.uniform(0, 10, len(tmp))
            df = df.append(tmp)
        df = df.pivot(index="timestamp", columns="segment")
        df = df.reorder_levels([1, 0], axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return TSDataset(df, freq="1D")

    df1 = generate_df(start_time="2020-01-01")
    df2 = generate_df(start_time="2019-01-01")

    return df1, df2


@pytest.fixture
def two_dfs_with_different_segments_sets(random_seed):
    """Generate two dataframes with the same timestamps and different segments"""

    def generate_df(n_segments):
        df = pd.DataFrame()
        for i in range(n_segments):
            tmp = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-01-01")})
            tmp["segment"] = f"segment_{i + 1}"
            tmp["target"] = np.random.uniform(0, 10, len(tmp))
            df = df.append(tmp)
        df = df.pivot(index="timestamp", columns="segment")
        df = df.reorder_levels([1, 0], axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return TSDataset(df, freq="1D")

    df1 = generate_df(n_segments=5)
    df2 = generate_df(n_segments=10)

    return df1, df2


@pytest.fixture
def train_test_dfs(random_seed):
    """Generate two dataframes with the same segments and the same timestamps"""

    def generate_df():
        df = pd.DataFrame()
        for i in range(5):
            tmp = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-01-01")})
            tmp["segment"] = f"segment_{i + 1}"
            tmp["target"] = np.random.uniform(0, 10, len(tmp))
            df = df.append(tmp)
        df = df.pivot(index="timestamp", columns="segment")
        df = df.reorder_levels([1, 0], axis=1)
        df = df.sort_index(axis=1)
        df.columns.names = ["segment", "feature"]
        return TSDataset(df, freq="1D")

    df1 = generate_df()
    df2 = generate_df()

    return df1, df2


@pytest.fixture
def simple_df() -> TSDataset:
    """Generate dataset with simple values without any noise"""
    history = 49

    df1 = pd.DataFrame()
    df1["target"] = np.arange(history)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-01-01", periods=history)

    df2 = pd.DataFrame()
    df2["target"] = [0, 2, 4, 6, 8, 10, 12] * 7
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-01-01", periods=history)

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="1d")

    return tsds


@pytest.fixture()
def outliers_df():
    timestamp1 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-01"))
    target1 = [np.sin(i) for i in range(len(timestamp1))]
    target1[10] += 10

    timestamp2 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-10"))
    target2 = [np.sin(i) for i in range(len(timestamp2))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12

    df1 = pd.DataFrame({"timestamp": timestamp1, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp2, "target": target2, "segment": "2"})

    df = pd.concat([df1, df2], ignore_index=True)
    return df


@pytest.fixture
def example_df_(random_seed) -> pd.DataFrame:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = ["segment_1"] * periods
    df1["target"] = np.random.uniform(10, 20, size=periods)
    df1["target_no_change"] = df1["target"]

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = ["segment_2"] * periods
    df2["target"] = np.random.uniform(-15, 5, size=periods)
    df2["target_no_change"] = df2["target"]

    df = pd.concat((df1, df2))
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


@pytest.fixture
def example_tsds(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds


@pytest.fixture
def example_reg_tsds(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(10, 20, size=periods)

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(-15, 5, size=periods)

    exog_weekend_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods + 7)})
    exog_weekend_1["segment"] = "segment_1"
    exog_weekend_1["regressor_exog_weekend"] = ((exog_weekend_1.timestamp.dt.dayofweek) // 5 == 1).astype("category")

    exog_weekend_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods + 7)})
    exog_weekend_2["segment"] = "segment_2"
    exog_weekend_2["regressor_exog_weekend"] = ((exog_weekend_2.timestamp.dt.dayofweek) // 5 == 1).astype("category")

    df = pd.concat([df1, df2]).reset_index(drop=True)
    exog = pd.concat([exog_weekend_1, exog_weekend_2]).reset_index(drop=True)

    df = TSDataset.to_dataset(df)
    exog = TSDataset.to_dataset(exog)

    tsds = TSDataset(df, freq="D", df_exog=exog, known_future="all")

    return tsds


@pytest.fixture()
def outliers_tsds():
    timestamp1 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-01"))
    target1 = [np.sin(i) for i in range(len(timestamp1))]
    target1[10] += 10

    timestamp2 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-10"))
    target2 = [np.sin(i) for i in range(len(timestamp2))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12

    df1 = pd.DataFrame({"timestamp": timestamp1, "target": target1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp2, "target": target2, "segment": "2"})

    df = pd.concat([df1, df2], ignore_index=True)

    df = df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]

    exog = df.copy()
    exog.columns.set_levels(["exog"], level="feature", inplace=True)

    tsds = TSDataset(df, "1d", exog)

    return tsds


@pytest.fixture
def outliers_df_with_two_columns() -> TSDataset:
    timestamp1 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-10"))
    target1 = [np.sin(i) for i in range(len(timestamp1))]
    feature1 = [np.cos(i) for i in range(len(timestamp1))]
    target1[10] += 10
    feature1[7] += 10

    timestamp2 = np.arange(np.datetime64("2021-01-01"), np.datetime64("2021-02-10"))
    target2 = [np.sin(i) for i in range(len(timestamp2))]
    feature2 = [np.cos(i) for i in range(len(timestamp2))]
    target2[8] += 8
    target2[15] = 2
    target2[26] -= 12
    feature2[25] += 10

    df1 = pd.DataFrame({"timestamp": timestamp1, "target": target1, "feature": feature1, "segment": "1"})
    df2 = pd.DataFrame({"timestamp": timestamp2, "target": target2, "feature": feature2, "segment": "2"})

    df = pd.concat([df1, df2], ignore_index=True)

    df = df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]

    tsds = TSDataset(df, "1d")

    return tsds


@pytest.fixture
def multitrend_df() -> pd.DataFrame:
    """Generate one segment pd.DataFrame with multiple linear trend."""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-05-31")})
    ns = [100, 150, 80, 187]
    ks = [0.4, -0.3, 0.8, -0.6]
    x = np.zeros(shape=(len(df)))
    left = 0
    right = 0
    for i, (n, k) in enumerate(zip(ns, ks)):
        right += n
        x[left:right] = np.arange(0, n, 1) * k
        for _n, _k in zip(ns[:i], ks[:i]):
            x[left:right] += _n * _k
        left = right
    df["target"] = x
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df=df)
    return df


@pytest.fixture
def ts_with_different_series_length(example_df: pd.DataFrame) -> TSDataset:
    """Generate TSDataset with different lengths series."""
    df = TSDataset.to_dataset(example_df)
    df.loc[:4, pd.IndexSlice["segment_1", "target"]] = None
    ts = TSDataset(df=df, freq="H")
    return ts


@pytest.fixture
def imbalanced_tsdf(random_seed) -> TSDataset:
    """Generate two series with big time range difference"""
    df1 = pd.DataFrame({"timestamp": pd.date_range("2021-01-25", "2021-02-01", freq="D")})
    df1["segment"] = "segment_1"
    df1["target"] = np.random.uniform(0, 5, len(df1))

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-02-01", freq="D")})
    df2["segment"] = "segment_2"
    df2["target"] = np.random.uniform(0, 5, len(df2))

    df = df1.append(df2)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def example_tsdf(random_seed) -> TSDataset:
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="H")
    return df


@pytest.fixture
def big_daily_example_tsdf(random_seed) -> TSDataset:
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2019-01-01", end="2020-04-01", freq="D")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2019-06-01", end="2020-04-01", freq="D")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="D")
    return df


@pytest.fixture
def big_example_tsdf(random_seed) -> TSDataset:
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2020-01-01", end="2021-02-01", freq="D")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2020-01-01", end="2021-02-01", freq="D")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.pivot(index="timestamp", columns="segment").reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    df = TSDataset(df, freq="D")
    return df


@pytest.fixture
def simple_df_relevance() -> Tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = pd.date_range("2021-01-01", "2021-02-01")

    df_1 = pd.DataFrame({"timestamp": timestamp, "target": np.arange(32), "segment": "1"})
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "target": np.arange(5, 32), "segment": "2"})
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(df)

    timestamp = pd.date_range("2020-12-01", "2021-02-11")
    regr1_2 = np.sin(-np.arange(len(timestamp) - 5))
    regr2_2 = np.log(np.arange(1, len(timestamp) - 4))
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "regressor_1": np.arange(len(timestamp)),
            "regressor_2": np.zeros(len(timestamp)),
            "segment": "1",
        }
    )
    df_2 = pd.DataFrame({"timestamp": timestamp[5:], "regressor_1": regr1_2, "regressor_2": regr2_2, "segment": "2"})
    df_exog = pd.concat([df_1, df_2], ignore_index=True)
    df_exog = TSDataset.to_dataset(df_exog)

    return df, df_exog
