import numpy as np
import pandas as pd
import pytest

from etna.models.moving_average import MovingAverageModel
from etna.models.naive import NaiveModel
from etna.models.seasonal_ma import SeasonalMovingAverageModel


@pytest.mark.parametrize("model", [SeasonalMovingAverageModel, NaiveModel, MovingAverageModel])
def test_simple_model_forecaster_run(simple_df, model):
    sma_model = model()
    sma_model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7)
    res = sma_model.forecast(future_ts)
    res = res.to_pandas(flatten=True)
    assert not res.isnull().values.any()
    assert len(res) == 14


def test_seasonal_moving_average_forecaster_correct(simple_df):
    model = SeasonalMovingAverageModel(window=3, seasonality=7)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = np.arange(35, 42)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-02-19", periods=7)

    df2 = pd.DataFrame()
    df2["target"] = [0, 2, 4, 6, 8, 10, 12]
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-02-19", periods=7)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])
    assert np.all(res.values == answer.values)


def test_naive_forecaster_correct(simple_df):
    model = NaiveModel(lag=3)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = [46, 47, 48] * 2 + [46]
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-02-19", periods=7)

    df2 = pd.DataFrame()
    df2["target"] = [8, 10, 12] * 2 + [8]
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-02-19", periods=7)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])

    assert np.all(res.values == answer.values)


def test_moving_average_forecaster_correct(simple_df):
    model = MovingAverageModel(window=5)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    tmp = np.arange(44, 49)
    for i in range(7):
        tmp = np.append(tmp, [tmp[-5:].mean()])
    df1["target"] = tmp[-7:]
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-02-19", periods=7)

    df2 = pd.DataFrame()
    tmp = np.arange(0, 13, 2)
    for i in range(7):
        tmp = np.append(tmp, [tmp[-5:].mean()])
    df2["target"] = tmp[-7:]
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-02-19", periods=7)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])

    assert np.all(res.values == answer.values)
