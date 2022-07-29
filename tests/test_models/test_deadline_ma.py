import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.models.deadline_ma import DeadlineMovingAverageModel


@pytest.fixture()
def df():
    """Generate dataset with simple values without any noise"""
    history = 140

    df1 = pd.DataFrame()
    df1["target"] = np.arange(history)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-01-01", periods=history)

    df2 = pd.DataFrame()
    df2["target"] = [0, 2, 4, 6, 8, 10, 12] * 20
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-01-01", periods=history)

    df = pd.concat([df1, df2]).reset_index(drop=True)
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="1d")

    return tsds


@pytest.mark.parametrize("model", [DeadlineMovingAverageModel])
def test_simple_model_forecaster_run(simple_df, model):
    sma_model = model(window=1)
    sma_model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7)
    res = sma_model.forecast(future_ts)
    res = res.to_pandas(flatten=True)
    assert not res.isnull().values.any()
    assert len(res) == 14


def test_deadline_moving_average_forecaster_correct(df):
    model = DeadlineMovingAverageModel(window=3, seasonality="month")
    model.fit(df)
    future_ts = df.make_future(future_steps=20)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = np.array(
        [
            79 + 2 / 3,
            80 + 2 / 3,
            81 + 2 / 3,
            82 + 2 / 3,
            83 + 2 / 3,
            84 + 2 / 3,
            85 + 2 / 3,
            86 + 2 / 3,
            87 + 2 / 3,
            88 + 2 / 3,
            89 + 1 / 3,
            89 + 2 / 3,
            90 + 2 / 3,
            91 + 2 / 3,
            92 + 2 / 3,
            93 + 2 / 3,
            94.0 + 2 / 3,
            95.0 + 2 / 3,
            96.0 + 2 / 3,
            97 + 2 / 3,
        ]
    )
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-05-20", periods=20)

    df2 = pd.DataFrame()
    df2["target"] = np.array(
        [
            5.0 + 1 / 3,
            7.0 + 1 / 3,
            4.0 + 2 / 3,
            6.0 + 2 / 3,
            8.0 + 2 / 3,
            6.0,
            3.0 + 1 / 3,
            5.0 + 1 / 3,
            7.0 + 1 / 3,
            4.0 + 2 / 3,
            6.0,
            6.0 + 2 / 3,
            4.0,
            6.0,
            8.0,
            5.0 + 1 / 3,
            7.0 + 1 / 3,
            4.0 + 2 / 3,
            6.0 + 2 / 3,
            4.0,
        ]
    )
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-05-20", periods=20)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])
    assert np.all(res.values == answer.values)
