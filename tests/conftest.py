from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from etna.datasets import generate_const_df
from etna.datasets.hierarchical_structure import HierarchicalStructure
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
        dfs = []
        for i in range(5):
            tmp = pd.DataFrame({"timestamp": pd.date_range(start_time, "2021-01-01")})
            tmp["segment"] = f"segment_{i + 1}"
            tmp["target"] = np.random.uniform(0, 10, len(tmp))
            dfs.append(tmp)
        df = pd.concat(dfs)
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
        dfs = []
        for i in range(n_segments):
            tmp = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-01-01")})
            tmp["segment"] = f"segment_{i + 1}"
            tmp["target"] = np.random.uniform(0, 10, len(tmp))
            dfs.append(tmp)
        df = pd.concat(dfs)
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
        dfs = []
        for i in range(5):
            tmp = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2021-01-01")})
            tmp["segment"] = f"segment_{i + 1}"
            tmp["target"] = np.random.uniform(0, 10, len(tmp))
            dfs.append(tmp)
        df = pd.concat(dfs)
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

    df = pd.concat((df1, df2))
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


@pytest.fixture
def const_ts_anomal() -> TSDataset:
    df = generate_const_df(periods=15, start_time="2020-01-01", scale=1.0, n_segments=2)
    ts = TSDataset(TSDataset.to_dataset(df), freq="D")
    return ts


@pytest.fixture
def ts_diff_endings(example_reg_tsds):
    ts = deepcopy(example_reg_tsds)
    ts.loc[ts.index[-5] :, pd.IndexSlice["segment_1", "target"]] = np.NAN
    return ts


@pytest.fixture
def ts_with_nans_in_tails(example_df):
    df = TSDataset.to_dataset(example_df)
    df.loc[:4, pd.IndexSlice["segment_1", "target"]] = None
    df.loc[-3:, pd.IndexSlice["segment_1", "target"]] = None
    ts = TSDataset(df, freq="H")
    return ts


@pytest.fixture
def ts_with_nans(ts_with_nans_in_tails):
    df = ts_with_nans_in_tails.to_pandas()
    df.loc[[df.index[5], df.index[8]], pd.IndexSlice["segment_1", "target"]] = None
    ts = TSDataset(df, freq="H")
    return ts


@pytest.fixture
def toy_dataset_equal_targets_and_quantiles():
    n_periods = 5
    n_segments = 2

    time = list(pd.date_range("2020-01-01", periods=n_periods, freq="1D"))

    df = {
        "timestamp": time * n_segments,
        "segment": ["a"] * n_periods + ["b"] * n_periods,
        "target": np.concatenate((np.array((2, 3, 4, 5, 5)), np.array((3, 3, 3, 5, 2)))).astype(np.float64),
        "target_0.01": np.concatenate((np.array((2, 3, 4, 5, 5)), np.array((3, 3, 3, 5, 2)))).astype(np.float64),
    }
    df = TSDataset.to_dataset(pd.DataFrame(df))
    ts = TSDataset(df, freq="D")
    return ts


@pytest.fixture
def toy_dataset_with_mean_shift_in_target():
    mean_1 = 10
    mean_2 = 20
    n_periods = 5
    n_segments = 2

    time = list(pd.date_range("2020-01-01", periods=n_periods, freq="1D"))

    df = {
        "timestamp": time * n_segments,
        "segment": ["a"] * n_periods + ["b"] * n_periods,
        "target": np.concatenate((np.array((-1, 3, 3, -4, -1)) + mean_1, np.array((-2, 3, -4, 5, -2)) + mean_2)).astype(
            np.float64
        ),
        "target_0.01": np.concatenate((np.array((-1, 3, 3, -4, -1)), np.array((-2, 3, -4, 5, -2)))).astype(np.float64),
    }
    df = TSDataset.to_dataset(pd.DataFrame(df))
    ts = TSDataset(df, freq="1D")
    return ts


@pytest.fixture
def hierarchical_structure():
    hs = HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]},
        level_names=["total", "market", "product"],
    )
    return hs


@pytest.fixture
def total_level_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"],
            "segment": ["total"] * 2,
            "target": [11.0, 22.0],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [1.0, 2.0] + [10.0, 20.0],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def product_level_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "segment": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1.0, 1.0] + [0.0, 1.0] + [3.0, 18.0] + [7.0, 2.0],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def product_level_df_w_nans():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] * 4,
            "segment": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4,
            "target": [None, 0, 1, 2] + [3, 4, 5, None] + [7, 8, None, 9] + [10, 11, 12, 13],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_df_w_nans():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] * 2,
            "segment": ["X"] * 4 + ["Y"] * 4,
            "target": [None, 4, 6, None] + [17, 19, None, 22],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def total_level_df_w_nans():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"],
            "segment": ["total"] * 4,
            "target": [None, 23, None, None],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def product_level_constant_hierarchical_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] * 4,
            "segment": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4,
            "target": [1, 1, 1, 1] + [2, 2, 2, 2] + [3, 3, 3, 3] + [4, 4, 4, 4],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_constant_hierarchical_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] * 2,
            "segment": ["X"] * 4 + ["Y"] * 4,
            "target": [3, 3, 3, 3] + [7, 7, 7, 7],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_constant_hierarchical_df_exog():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"] * 2,
            "segment": ["X"] * 6 + ["Y"] * 6,
            "regressor": [1, 1, 1, 1, 1, 1] * 2,
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def total_level_simple_hierarchical_ts(total_level_df, hierarchical_structure):
    ts = TSDataset(df=total_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_simple_hierarchical_ts(market_level_df, hierarchical_structure):
    ts = TSDataset(df=market_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def product_level_simple_hierarchical_ts(product_level_df, hierarchical_structure):
    ts = TSDataset(df=product_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def simple_no_hierarchy_ts(market_level_df):
    ts = TSDataset(df=market_level_df, freq="D")
    return ts


@pytest.fixture
def market_level_constant_hierarchical_ts(market_level_constant_hierarchical_df, hierarchical_structure):
    ts = TSDataset(df=market_level_constant_hierarchical_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_constant_hierarchical_ts_w_exog(
    market_level_constant_hierarchical_df, market_level_constant_hierarchical_df_exog, hierarchical_structure
):
    ts = TSDataset(
        df=market_level_constant_hierarchical_df,
        df_exog=market_level_constant_hierarchical_df_exog,
        freq="D",
        hierarchical_structure=hierarchical_structure,
        known_future="all",
    )
    return ts


@pytest.fixture
def product_level_constant_hierarchical_ts(product_level_constant_hierarchical_df, hierarchical_structure):
    ts = TSDataset(
        df=product_level_constant_hierarchical_df,
        freq="D",
        hierarchical_structure=hierarchical_structure,
    )
    return ts


@pytest.fixture
def product_level_constant_hierarchical_ts_with_exog(
    product_level_constant_hierarchical_df, market_level_constant_hierarchical_df_exog, hierarchical_structure
):
    ts = TSDataset(
        df=product_level_constant_hierarchical_df,
        df_exog=market_level_constant_hierarchical_df_exog,
        freq="D",
        hierarchical_structure=hierarchical_structure,
        known_future="all",
    )
    return ts


@pytest.fixture
def product_level_constant_forecast_with_quantiles(hierarchical_structure):
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"] * 4,
            "segment": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1, 1] + [2, 2] + [3, 3] + [4, 4],
            "target_0.25": [1 / 2, 1 / 4] + [1, 1 / 2] + [2, 1] + [3, 2],
            "target_0.75": [2, 3] + [3, 4] + [4, 5] + [5, 6],
        },
        dtype=float,
    )
    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def product_level_constant_forecast_with_target_components(hierarchical_structure):
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"] * 4,
            "segment": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1, 1] + [2, 2] + [3, 3] + [4, 4],
        },
        dtype=float,
    )
    target_components_df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"] * 4,
            "segment": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target_component_a": [0.7, 0.7] + [1.5, 1.5] + [2, 2] + [3, 3],
            "target_component_b": [0.3, 0.3] + [0.5, 0.5] + [1, 1] + [1, 1],
        },
        dtype=float,
    )
    df = TSDataset.to_dataset(df=df)
    target_components_df = TSDataset.to_dataset(target_components_df)
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    ts.add_target_components(target_components_df=target_components_df)
    return ts


@pytest.fixture
def market_level_constant_forecast_with_quantiles(hierarchical_structure):
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [3, 3] + [7, 7],
            "target_0.25": [1.5, 0.75] + [5, 3],
            "target_0.75": [5, 7] + [9, 11],
        },
        dtype=float,
    )
    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_constant_forecast_with_target_components(hierarchical_structure):
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [3, 3] + [7, 7],
        },
        dtype=float,
    )
    target_components_df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target_component_a": [2.2, 2.2] + [5, 5],
            "target_component_b": [0.8, 0.8] + [2, 2],
        },
        dtype=float,
    )
    df = TSDataset.to_dataset(df=df)
    target_components_df = TSDataset.to_dataset(target_components_df)
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    ts.add_target_components(target_components_df=target_components_df)
    return ts


@pytest.fixture
def total_level_constant_forecast_with_quantiles(hierarchical_structure):
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"],
            "segment": ["total"] * 2,
            "target": [10, 10],
            "target_0.25": [6.5, 3.75],
            "target_0.75": [14, 18],
        },
        dtype=float,
    )
    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def total_level_constant_forecast_with_target_components(hierarchical_structure):
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"],
            "segment": ["total"] * 2,
            "target": [10, 10],
        },
        dtype=float,
    )
    target_components_df = pd.DataFrame(
        {
            "timestamp": ["2000-01-05", "2000-01-06"],
            "segment": ["total"] * 2,
            "target_component_a": [7.2, 7.2],
            "target_component_b": [2.8, 2.8],
        },
        dtype=float,
    )
    df = TSDataset.to_dataset(df=df)
    target_components_df = TSDataset.to_dataset(target_components_df)
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    ts.add_target_components(target_components_df=target_components_df)
    return ts
