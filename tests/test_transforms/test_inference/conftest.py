import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data


@pytest.fixture
def regular_ts(random_seed) -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_1["segment"] = "segment_1"
    df_1["target"] = np.random.uniform(10, 20, size=periods)

    df_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_2["segment"] = "segment_2"
    df_2["target"] = np.random.uniform(-15, 5, size=periods)

    df_3 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df_3["segment"] = "segment_3"
    df_3["target"] = np.random.uniform(-5, 5, size=periods)

    df = pd.concat([df_1, df_2, df_3]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")

    return tsds


@pytest.fixture
def ts_with_exog(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas(flatten=True)
    periods = 200
    timestamp = pd.date_range("2020-01-01", periods=periods)
    df_exog_common = pd.DataFrame(
        {
            "timestamp": timestamp,
            "positive": 1,
            "weekday": timestamp.weekday,
            "monthday": timestamp.day,
            "month": timestamp.month,
            "year": timestamp.year,
        }
    )
    df_exog_wide = duplicate_data(df=df_exog_common, segments=regular_ts.segments)

    rng = np.random.default_rng(1)
    df_exog_wide.loc[:, pd.IndexSlice["segment_1", "positive"]] = rng.uniform(5, 10, size=periods)
    df_exog_wide.loc[:, pd.IndexSlice["segment_2", "positive"]] = rng.uniform(5, 10, size=periods)
    df_exog_wide.loc[:, pd.IndexSlice["segment_3", "positive"]] = rng.uniform(5, 10, size=periods)

    ts = TSDataset(df=TSDataset.to_dataset(df).iloc[5:], df_exog=df_exog_wide, freq="D")
    return ts


@pytest.fixture
def positive_ts() -> TSDataset:
    periods = 100
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq="D")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq="D")})
    df_3 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2020-01-01", periods=periods, freq="D")})
    generator = np.random.RandomState(seed=1)

    df_1["segment"] = "segment_1"
    df_1["target"] = np.abs(generator.normal(loc=10, scale=1, size=len(df_1))) + 1

    df_2["segment"] = "segment_2"
    df_2["target"] = np.abs(generator.normal(loc=20, scale=1, size=len(df_2))) + 1

    df_3["segment"] = "segment_3"
    df_3["target"] = np.abs(generator.normal(loc=30, scale=1, size=len(df_2))) + 1

    classic_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
    wide_df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=wide_df, freq="D")
    return ts


@pytest.fixture
def ts_to_fill(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas()
    df.iloc[5, 0] = np.NaN
    df.iloc[10, 1] = np.NaN
    df.iloc[20, 2] = np.NaN
    df.iloc[-5, 0] = np.NaN
    df.iloc[-10, 1] = np.NaN
    df.iloc[-20, 2] = np.NaN
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def ts_to_resample() -> TSDataset:
    df_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=120),
            "segment": "segment_1",
            "target": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=120),
            "segment": "segment_2",
            "target": ([1] + 23 * [0]) * 5,
        }
    )
    df_3 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="H", periods=120),
            "segment": "segment_3",
            "target": ([4] + 23 * [0]) * 5,
        }
    )
    df = pd.concat([df_1, df_2, df_3], ignore_index=True)

    df_exog_1 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=8),
            "segment": "segment_1",
            "regressor_exog": 2,
        }
    )
    df_exog_2 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=8),
            "segment": "segment_2",
            "regressor_exog": 40,
        }
    )
    df_exog_3 = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-05", freq="D", periods=8),
            "segment": "segment_3",
            "regressor_exog": 40,
        }
    )
    df_exog = pd.concat([df_exog_1, df_exog_2, df_exog_3], ignore_index=True)
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="H", df_exog=TSDataset.to_dataset(df_exog), known_future="all")
    return ts


@pytest.fixture
def ts_with_outliers(regular_ts) -> TSDataset:
    df = regular_ts.to_pandas()
    df.iloc[5, 0] *= 100
    df.iloc[10, 1] *= 100
    df.iloc[20, 2] *= 100
    df.iloc[-5, 0] *= 100
    df.iloc[-10, 1] *= 100
    df.iloc[-20, 2] *= 100
    ts = TSDataset(df=df, freq="D")
    return ts
