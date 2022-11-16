import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models.deadline_ma import DeadlineMovingAverageModel
from etna.models.deadline_ma import SeasonalityMode
from etna.models.deadline_ma import _DeadlineMovingAverageModel
from etna.models.moving_average import MovingAverageModel
from etna.models.naive import NaiveModel
from etna.models.seasonal_ma import SeasonalMovingAverageModel
from etna.models.seasonal_ma import _SeasonalMovingAverageModel
from etna.pipeline import Pipeline


def _check_forecast(ts, model, horizon):
    model.fit(ts)
    future_ts = ts.make_future(future_steps=horizon, tail_steps=model.context_size)
    res = model.forecast(ts=future_ts, prediction_size=horizon)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == horizon * 2


def _check_predict(ts, model, prediction_size):
    model.fit(ts)
    res = model.predict(ts, prediction_size=prediction_size)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == prediction_size * 2


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


@pytest.mark.parametrize("model", [SeasonalMovingAverageModel, NaiveModel, MovingAverageModel])
def test_sma_model_forecast(simple_df, model):
    _check_forecast(ts=simple_df, model=model(), horizon=7)


@pytest.mark.parametrize("model", [SeasonalMovingAverageModel, NaiveModel, MovingAverageModel])
def test_sma_model_predict(simple_df, model):
    _check_predict(ts=simple_df, model=model(), prediction_size=7)


def test_sma_model_forecast_fail_not_enough_context(simple_df):
    sma_model = SeasonalMovingAverageModel(window=1000, seasonality=7)
    sma_model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=sma_model.context_size)
    with pytest.raises(ValueError, match="Given context isn't big enough"):
        _ = sma_model.forecast(future_ts, prediction_size=7)


def test_sma_model_predict_fail_not_enough_context(simple_df):
    sma_model = SeasonalMovingAverageModel(window=1000, seasonality=7)
    sma_model.fit(simple_df)
    with pytest.raises(ValueError, match="Given context isn't big enough"):
        _ = sma_model.predict(simple_df, prediction_size=7)


def test_sma_model_forecast_fail_nans_in_context(simple_df):
    sma_model = SeasonalMovingAverageModel(window=2, seasonality=2)
    sma_model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=sma_model.context_size)
    future_ts.df.iloc[1, 0] = np.NaN
    with pytest.raises(ValueError, match="There are NaNs in a forecast context"):
        _ = sma_model.forecast(future_ts, prediction_size=7)


def test_sma_model_predict_fail_nans_in_context(simple_df):
    sma_model = SeasonalMovingAverageModel(window=2, seasonality=7)
    sma_model.fit(simple_df)
    simple_df.df.iloc[-1, 0] = np.NaN
    with pytest.raises(ValueError, match="There are NaNs in a target column"):
        _ = sma_model.predict(simple_df, prediction_size=7)


@pytest.mark.parametrize(
    "freq, periods, start, prediction_size, seasonality, window, expected",
    [
        ("D", 31 + 1, "2020-01-01", 1, SeasonalityMode.month, 1, pd.Timestamp("2020-01-01")),
        ("D", 31 + 2, "2020-01-01", 1, SeasonalityMode.month, 1, pd.Timestamp("2020-01-02")),
        ("D", 31 + 5, "2020-01-01", 5, SeasonalityMode.month, 1, pd.Timestamp("2020-01-01")),
        ("D", 31 + 29 + 1, "2020-01-01", 1, SeasonalityMode.month, 2, pd.Timestamp("2020-01-01")),
        ("D", 31 + 29 + 31 + 1, "2020-01-01", 1, SeasonalityMode.month, 3, pd.Timestamp("2020-01-01")),
        ("H", 31 * 24 + 1, "2020-01-01", 1, SeasonalityMode.month, 1, pd.Timestamp("2020-01-01")),
        ("H", 31 * 24 + 2, "2020-01-01", 1, SeasonalityMode.month, 1, pd.Timestamp("2020-01-01 01:00")),
        ("H", 31 * 24 + 5, "2020-01-01", 5, SeasonalityMode.month, 1, pd.Timestamp("2020-01-01")),
        ("H", (31 + 29) * 24 + 1, "2020-01-01", 1, SeasonalityMode.month, 2, pd.Timestamp("2020-01-01")),
        ("H", (31 + 29 + 31) * 24 + 1, "2020-01-01", 1, SeasonalityMode.month, 3, pd.Timestamp("2020-01-01")),
        ("D", 366 + 1, "2020-01-01", 1, SeasonalityMode.year, 1, pd.Timestamp("2020-01-01")),
        ("D", 366 + 2, "2020-01-01", 1, SeasonalityMode.year, 1, pd.Timestamp("2020-01-02")),
        ("D", 366 + 5, "2020-01-01", 5, SeasonalityMode.year, 1, pd.Timestamp("2020-01-01")),
        ("D", 366 + 365 + 1, "2020-01-01", 1, SeasonalityMode.year, 2, pd.Timestamp("2020-01-01")),
        ("D", 366 + 365 + 365 + 1, "2020-01-01", 1, SeasonalityMode.year, 3, pd.Timestamp("2020-01-01")),
        ("H", 366 * 24 + 1, "2020-01-01", 1, SeasonalityMode.year, 1, pd.Timestamp("2020-01-01")),
        ("H", 366 * 24 + 2, "2020-01-01", 1, SeasonalityMode.year, 1, pd.Timestamp("2020-01-01 01:00")),
        ("H", 366 * 24 + 5, "2020-01-01", 5, SeasonalityMode.year, 1, pd.Timestamp("2020-01-01")),
        ("H", (366 + 365) * 24 + 1, "2020-01-01", 1, SeasonalityMode.year, 2, pd.Timestamp("2020-01-01")),
        ("H", (366 + 365 + 365) * 24 + 1, "2020-01-01", 1, SeasonalityMode.year, 3, pd.Timestamp("2020-01-01")),
    ],
)
def test_deadline_get_context_beginning_ok(freq, periods, start, prediction_size, seasonality, window, expected):
    df = pd.DataFrame({"timestamp": pd.date_range(start=start, periods=periods, freq=freq)})

    obtained = _DeadlineMovingAverageModel._get_context_beginning(df, prediction_size, seasonality, window)

    assert obtained == expected


@pytest.mark.parametrize(
    "freq, periods, start, prediction_size, seasonality, window",
    [
        ("D", 1, "2020-01-01", 1, SeasonalityMode.month, 1),
        ("H", 1, "2020-01-01", 1, SeasonalityMode.month, 1),
        ("D", 1, "2020-01-01", 1, SeasonalityMode.year, 1),
        ("H", 1, "2020-01-01", 1, SeasonalityMode.year, 1),
        ("D", 1, "2020-01-01", 2, SeasonalityMode.month, 1),
        ("H", 1, "2020-01-01", 2, SeasonalityMode.month, 1),
        ("D", 1, "2020-01-01", 2, SeasonalityMode.year, 1),
        ("H", 1, "2020-01-01", 2, SeasonalityMode.year, 1),
        ("D", 31 + 1, "2020-01-01", 2, SeasonalityMode.month, 1),
        ("H", 31 * 24 + 1, "2020-01-01", 2, SeasonalityMode.month, 1),
        ("D", 366 + 1, "2020-01-01", 2, SeasonalityMode.year, 1),
        ("H", 366 * 24 + 1, "2020-01-01", 2, SeasonalityMode.year, 1),
    ],
)
def test_deadline_get_context_beginning_fail_not_enough_context(
    freq, periods, start, prediction_size, seasonality, window
):
    df = pd.DataFrame({"timestamp": pd.date_range(start=start, periods=periods, freq=freq)})

    with pytest.raises(ValueError, match="Given context isn't big enough"):
        _ = _DeadlineMovingAverageModel._get_context_beginning(df, prediction_size, seasonality, window)


@pytest.mark.parametrize("model", [DeadlineMovingAverageModel])
def test_deadline_model_forecast(simple_df, model):
    _check_forecast(ts=simple_df, model=model(window=1), horizon=7)


@pytest.mark.parametrize("model", [DeadlineMovingAverageModel])
def test_deadline_model_predict(simple_df, model):
    _check_predict(ts=simple_df, model=model(window=1), prediction_size=7)


def test_deadline_model_forecast_fail_not_enough_context(simple_df):
    model = DeadlineMovingAverageModel(window=1000)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=model.context_size)
    with pytest.raises(ValueError, match="Given context isn't big enough"):
        _ = model.forecast(future_ts, prediction_size=7)


def test_deadline_model_predict_fail_not_enough_context(simple_df):
    model = DeadlineMovingAverageModel(window=1000)
    model.fit(simple_df)
    with pytest.raises(ValueError, match="Given context isn't big enough"):
        _ = model.forecast(simple_df, prediction_size=7)


def test_deadline_model_forecast_fail_nans_in_context(simple_df):
    model = DeadlineMovingAverageModel(window=1)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=model.context_size)
    future_ts.df.iloc[1, 0] = np.NaN
    with pytest.raises(ValueError, match="There are NaNs in a forecast context"):
        _ = model.forecast(future_ts, prediction_size=7)


def test_deadline_model_predict_fail_nans_in_context(simple_df):
    model = DeadlineMovingAverageModel(window=1)
    model.fit(simple_df)
    simple_df.df.iloc[-1, 0] = np.NaN
    with pytest.raises(ValueError, match="There are NaNs in a target column"):
        _ = model.predict(simple_df, prediction_size=7)


def test_seasonal_moving_average_forecast_correct(simple_df):
    model = SeasonalMovingAverageModel(window=3, seasonality=7)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=model.context_size)
    res = model.forecast(future_ts, prediction_size=7)
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


def test_naive_forecast_correct(simple_df):
    model = NaiveModel(lag=3)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=model.context_size)
    res = model.forecast(future_ts, prediction_size=7)
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


def test_moving_average_forecast_correct(simple_df):
    model = MovingAverageModel(window=5)
    model.fit(simple_df)
    future_ts = simple_df.make_future(future_steps=7, tail_steps=model.context_size)
    res = model.forecast(future_ts, prediction_size=7)
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


def test_deadline_moving_average_forecast_correct(df):
    model = DeadlineMovingAverageModel(window=3, seasonality="month")
    model.fit(df)
    future_ts = df.make_future(future_steps=20, tail_steps=model.context_size)
    res = model.forecast(future_ts, prediction_size=20)
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


def test_seasonal_moving_average_predict_correct(simple_df):
    model = SeasonalMovingAverageModel(window=2, seasonality=2)
    model.fit(simple_df)
    res = model.predict(simple_df, prediction_size=7)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = np.arange(39, 46)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-02-12", periods=7)

    df2 = pd.DataFrame()
    df2["target"] = [8, 10, 5, 7, 2, 4, 6]
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-02-12", periods=7)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])
    assert np.all(res.values == answer.values)


def test_naive_predict_correct(simple_df):
    model = NaiveModel(lag=3)
    model.fit(simple_df)
    res = model.predict(simple_df, prediction_size=7)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = np.arange(39, 46)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-02-12", periods=7)

    df2 = pd.DataFrame()
    df2["target"] = [8, 10, 12, 0, 2, 4, 6]
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-02-12", periods=7)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])

    assert np.all(res.values == answer.values)


def test_moving_average_predict_correct(simple_df):
    model = MovingAverageModel(window=5)
    model.fit(simple_df)
    res = model.predict(simple_df, prediction_size=7)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = np.arange(39, 46)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-02-12", periods=7)

    df2 = pd.DataFrame()
    df2["target"] = [8, 7.2, 6.4, 5.6, 4.8, 4.0, 6.0]
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-02-12", periods=7)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])

    assert np.all(res.values == answer.values)


def test_deadline_moving_average_predict_correct(df):
    model = DeadlineMovingAverageModel(window=3, seasonality="month")
    model.fit(df)
    res = model.predict(df, prediction_size=20)
    res = res.to_pandas(flatten=True)[["target", "segment", "timestamp"]]

    df1 = pd.DataFrame()
    df1["target"] = np.array(
        [
            59,
            60 + 2 / 3,
            61 + 2 / 3,
            62 + 2 / 3,
            63 + 2 / 3,
            64 + 2 / 3,
            65 + 2 / 3,
            66 + 2 / 3,
            67 + 2 / 3,
            68 + 2 / 3,
            69 + 2 / 3,
            70 + 2 / 3,
            71 + 2 / 3,
            72 + 2 / 3,
            73 + 2 / 3,
            74 + 2 / 3,
            75 + 2 / 3,
            76 + 2 / 3,
            77 + 2 / 3,
            78 + 2 / 3,
        ]
    )
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-04-30", periods=20)

    df2 = pd.DataFrame()
    df2["target"] = np.array(
        [
            6,
            4 + 2 / 3,
            6 + 2 / 3,
            8 + 2 / 3,
            6,
            3 + 1 / 3,
            5 + 1 / 3,
            7 + 1 / 3,
            4 + 2 / 3,
            6 + 2 / 3,
            8 + 2 / 3,
            6,
            3 + 1 / 3,
            5 + 1 / 3,
            7 + 1 / 3,
            4 + 2 / 3,
            6 + 2 / 3,
            8 + 2 / 3,
            6,
            3 + 1 / 3,
        ]
    )
    df2["segment"] = "B"
    df2["timestamp"] = pd.date_range(start="2020-04-30", periods=20)

    answer = pd.concat([df2, df1], axis=0, ignore_index=True)
    res = res.sort_values(by=["segment", "timestamp"])
    answer = answer.sort_values(by=["segment", "timestamp"])
    assert np.all(res.values == answer.values)


@pytest.mark.parametrize(
    "model",
    [
        SeasonalMovingAverageModel(window=1, seasonality=1),
        SeasonalMovingAverageModel(window=3, seasonality=1),
        SeasonalMovingAverageModel(window=1, seasonality=3),
        SeasonalMovingAverageModel(window=3, seasonality=7),
        MovingAverageModel(window=3),
        NaiveModel(lag=5),
    ],
)
def test_context_size_seasonal_ma(model):
    expected_context_size = model.window * model.seasonality
    assert model.context_size == expected_context_size


@pytest.mark.parametrize(
    "model, freq, expected_context_size",
    [
        (DeadlineMovingAverageModel(window=1, seasonality="month"), "D", 31),
        (DeadlineMovingAverageModel(window=3, seasonality="month"), "D", 3 * 31),
        (DeadlineMovingAverageModel(window=1, seasonality="year"), "D", 366),
        (DeadlineMovingAverageModel(window=3, seasonality="year"), "D", 3 * 366),
        (DeadlineMovingAverageModel(window=1, seasonality="month"), "H", 31 * 24),
        (DeadlineMovingAverageModel(window=3, seasonality="month"), "H", 3 * 31 * 24),
        (DeadlineMovingAverageModel(window=1, seasonality="year"), "H", 366 * 24),
        (DeadlineMovingAverageModel(window=3, seasonality="year"), "H", 3 * 366 * 24),
    ],
)
def test_context_size_deadline_ma(model, freq, expected_context_size):
    # create dataframe
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=expected_context_size + 10, freq=freq),
            "segment": "segment_0",
            "target": 1,
        }
    )
    ts = TSDataset(df=TSDataset.to_dataset(df), freq=freq)

    # fit model
    model.fit(ts)

    # check result
    assert model.context_size == expected_context_size


@pytest.mark.parametrize(
    "etna_model_class",
    (SeasonalMovingAverageModel, MovingAverageModel, NaiveModel, DeadlineMovingAverageModel),
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
        (DeadlineMovingAverageModel, _DeadlineMovingAverageModel),
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


@pytest.fixture
def big_ts() -> TSDataset:
    np.random.seed(42)
    periods = 1000
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


def test_pipeline_with_deadline_model(big_ts):
    model = DeadlineMovingAverageModel(window=5, seasonality="month")
    pipeline = Pipeline(model=model, horizon=200)
    metrics, forecast, _ = pipeline.backtest(ts=big_ts, metrics=[MAE()], n_folds=3)
    assert not forecast.isnull().values.any()


@pytest.fixture()
def two_month_ts():
    history = 61

    df1 = pd.DataFrame()
    df1["target"] = np.arange(history)
    df1["segment"] = "A"
    df1["timestamp"] = pd.date_range(start="2020-01-01", periods=history)

    df = TSDataset.to_dataset(df1)
    tsds = TSDataset(df, freq="1d")
    return tsds


def test_deadline_model_forecast_correct_with_big_horizons(two_month_ts):
    model = DeadlineMovingAverageModel(window=2, seasonality="month")
    model.fit(two_month_ts)
    future_ts = two_month_ts.make_future(future_steps=90, tail_steps=model.context_size)
    res = model.forecast(future_ts, prediction_size=90)
    expected = np.array(
        [
            [16.5],
            [17.5],
            [18.5],
            [19.5],
            [20.5],
            [21.5],
            [22.5],
            [23.5],
            [24.5],
            [25.5],
            [26.5],
            [27.5],
            [28.5],
            [29.5],
            [30.5],
            [31.5],
            [32.5],
            [33.5],
            [34.5],
            [35.5],
            [36.5],
            [37.5],
            [38.5],
            [39.5],
            [40.5],
            [41.5],
            [42.5],
            [43.5],
            [44.0],
            [44.5],
            [45.5],
            [24.25],
            [25.25],
            [26.25],
            [27.25],
            [28.25],
            [29.25],
            [30.25],
            [31.25],
            [32.25],
            [33.25],
            [34.25],
            [35.25],
            [36.25],
            [37.25],
            [38.25],
            [39.25],
            [40.25],
            [41.25],
            [42.25],
            [43.25],
            [44.25],
            [45.25],
            [46.25],
            [47.25],
            [48.25],
            [49.25],
            [50.25],
            [51.25],
            [51.5],
            [52.75],
            [20.375],
            [21.375],
            [22.375],
            [23.375],
            [24.375],
            [25.375],
            [26.375],
            [27.375],
            [28.375],
            [29.375],
            [30.375],
            [31.375],
            [32.375],
            [33.375],
            [34.375],
            [35.375],
            [36.375],
            [37.375],
            [38.375],
            [39.375],
            [40.375],
            [41.375],
            [42.375],
            [43.375],
            [44.375],
            [45.375],
            [46.375],
            [47.375],
            [47.75],
        ]
    )
    assert np.all(res.df.values == expected)
