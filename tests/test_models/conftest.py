from copy import deepcopy

import numpy as np
import pytest

from etna.datasets import generate_ar_df
from etna.datasets.tsdataset import TSDataset


@pytest.fixture()
def new_format_df():
    classic_df = generate_ar_df(periods=30, start_time="2021-06-01", n_segments=2)
    df = TSDataset.to_dataset(classic_df)
    return df


@pytest.fixture()
def new_format_exog():
    exog = generate_ar_df(periods=60, start_time="2021-06-01", n_segments=2)
    df = TSDataset.to_dataset(exog)
    return df


@pytest.fixture()
def dfs_w_exog():
    df = generate_ar_df(start_time="2021-01-01", periods=105, n_segments=1)
    df["f1"] = np.sin(df["target"])
    df["f2"] = np.cos(df["target"])

    df.drop(columns=["segment"], inplace=True)
    train = df.iloc[:-5]
    test = df.iloc[-5:]
    return train, test


@pytest.fixture
def ts_with_non_convertable_category_regressor(example_tsds) -> TSDataset:
    ts = example_tsds
    df = ts.to_pandas(flatten=True)
    df_exog = deepcopy(df)
    df_exog["cat"] = "a"
    df_exog["cat"] = df_exog["cat"].astype("category")
    df_exog.drop(columns=["target"], inplace=True)
    df_wide = TSDataset.to_dataset(df).iloc[:-10]
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=ts.freq, known_future="all")
    return ts


@pytest.fixture
def ts_with_short_regressor(example_tsds) -> TSDataset:
    ts = example_tsds
    df = ts.to_pandas(flatten=True)
    df_exog = deepcopy(df)
    df_exog["exog"] = 1
    df_exog.drop(columns=["target"], inplace=True)
    df_wide = TSDataset.to_dataset(df).iloc[:-3]
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=ts.freq, known_future="all")
    return ts


@pytest.fixture
def ts_with_non_regressor_exog(example_tsds) -> TSDataset:
    ts = example_tsds
    df = ts.to_pandas(flatten=True)
    df_exog = deepcopy(df)
    df_exog["exog"] = 1
    df_exog.drop(columns=["target"], inplace=True)
    df_wide = TSDataset.to_dataset(df)
    df_exog_wide = TSDataset.to_dataset(df_exog)
    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq=ts.freq)
    return ts
