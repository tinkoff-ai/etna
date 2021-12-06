import numpy as np
import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.models import CatBoostModelMultiSegment
from etna.models import CatBoostModelPerSegment
from etna.transforms.lags import LagTransform


@pytest.mark.parametrize("catboostmodel", [CatBoostModelMultiSegment, CatBoostModelPerSegment])
def test_run(catboostmodel, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")

    lags = LagTransform(lags=[3, 4, 5], in_column="target")

    ts.fit_transform([lags])

    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


@pytest.mark.parametrize("catboostmodel", [CatBoostModelMultiSegment, CatBoostModelPerSegment])
def test_run_with_reg(catboostmodel, new_format_df, new_format_exog):
    df = new_format_df
    exog = new_format_exog
    exog.columns.set_levels(["regressor_exog"], level="feature", inplace=True)

    ts = TSDataset(df, "1d", df_exog=exog, known_future=["regressor_exog"])

    lags = LagTransform(lags=[3, 4, 5], in_column="target")
    lags_exog = LagTransform(lags=[3, 4, 5, 6], in_column="regressor_exog")

    ts.fit_transform([lags, lags_exog])

    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


@pytest.fixture
def constant_ts(size=40) -> TSDataset:
    constants = [7, 50, 130, 277, 370, 513]
    segments = [constant for constant in constants for _ in range(size)]
    ts_range = list(pd.date_range("2020-01-03", freq="D", periods=size))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * len(constants),
            "target": segments,
            "segment": [f"segment_{i+1}" for i in range(len(constants)) for _ in range(size)],
        }
    )
    ts = TSDataset(TSDataset.to_dataset(df), "D")
    train, test = ts.train_test_split(test_size=5)
    return train, test


def test_catboost_multi_segment_forecast(constant_ts):
    train, test = constant_ts
    horizon = len(test.df)

    lags = LagTransform(in_column="target", lags=[10, 11, 12])
    train.fit_transform([lags])
    future = train.make_future(horizon)

    model = CatBoostModelMultiSegment()
    model.fit(train)
    forecast = model.forecast(future)

    for segment in forecast.segments:
        assert np.allclose(test[:, segment, "target"], forecast[:, segment, "target"])
