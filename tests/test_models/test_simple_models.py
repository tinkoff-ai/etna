import numpy as np
import pandas as pd
import pytest

from etna.models.moving_average import MovingAverageModel
from etna.models.naive import NaiveModel
from etna.models.seasonal_ma import SeasonalMovingAverageModel
from etna.models.seasonal_ma import _SeasonalMovingAverageModel
from etna.pipeline import Pipeline


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


@pytest.mark.parametrize(
    "etna_model_class",
    (
        SeasonalMovingAverageModel,
        MovingAverageModel,
        NaiveModel,
    ),
)
def test_get_model_before_training(etna_model_class):
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = etna_model_class()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


@pytest.mark.parametrize(
    "etna_model_class,expected_class",
    (
        (NaiveModel, _SeasonalMovingAverageModel),
        (SeasonalMovingAverageModel, _SeasonalMovingAverageModel),
        (MovingAverageModel, _SeasonalMovingAverageModel),
    ),
)
def test_get_model_after_training(example_tsds, etna_model_class, expected_class):
    """Check that get_model method returns dict of objects of _SeasonalMovingAverageModel class."""
    pipeline = Pipeline(model=etna_model_class())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], expected_class)
