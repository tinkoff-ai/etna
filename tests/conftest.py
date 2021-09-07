import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset


@pytest.fixture()
def example_df():
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
def two_dfs_with_different_timestamps():
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
def two_dfs_with_different_segments_sets():
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
def train_test_dfs():
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
def example_df_() -> pd.DataFrame:
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
def example_tsds() -> TSDataset:
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
def example_reg_tsds() -> TSDataset:
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

    tsds = TSDataset(df, freq="D", df_exog=exog)

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
    exog.columns = pd.MultiIndex.from_arrays([["1", "2"], ["exog", "exog"]])

    tsds = TSDataset(df, "1d", exog)

    return tsds
