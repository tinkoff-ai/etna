from unittest.mock import Mock

import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_from_patterns_df
from etna.ensembles import DirectEnsemble
from etna.models import NaiveModel
from etna.pipeline import Pipeline


@pytest.fixture
def simple_ts_train():
    df = generate_from_patterns_df(patterns=[[1, 3, 5], [2, 4, 6], [7, 9, 11]], periods=3, start_time="2000-01-01")
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


@pytest.fixture
def simple_ts_forecast():
    df = generate_from_patterns_df(patterns=[[5, 3], [6, 4], [11, 9]], periods=2, start_time="2000-01-04")
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="D")
    return ts


def test_get_horizon():
    ensemble = DirectEnsemble(pipelines=[Mock(horizon=1), Mock(horizon=2)])
    assert ensemble.horizon == 2


def test_get_horizon_raise_error_on_same_horizons():
    with pytest.raises(ValueError, match="All the pipelines should have pairwise different horizons."):
        _ = DirectEnsemble(pipelines=[Mock(horizon=1), Mock(horizon=1)])


def test_forecast(simple_ts_train, simple_ts_forecast):
    ensemble = DirectEnsemble(
        pipelines=[
            Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=1),
            Pipeline(model=NaiveModel(lag=3), transforms=[], horizon=2),
        ]
    )
    ensemble.fit(simple_ts_train)
    forecast = ensemble.forecast()
    pd.testing.assert_frame_equal(forecast.to_pandas(), simple_ts_forecast.to_pandas())


def test_predict(simple_ts_train):
    ensemble = DirectEnsemble(
        pipelines=[
            Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=1),
            Pipeline(model=NaiveModel(lag=3), transforms=[], horizon=2),
        ]
    )
    smallest_pipeline = Pipeline(model=NaiveModel(lag=1), transforms=[], horizon=1)
    ensemble.fit(simple_ts_train)
    smallest_pipeline.fit(simple_ts_train)
    prediction = ensemble.predict(
        ts=simple_ts_train, start_timestamp=simple_ts_train.index[1], end_timestamp=simple_ts_train.index[2]
    )
    expected_prediction = smallest_pipeline.predict(
        ts=simple_ts_train, start_timestamp=simple_ts_train.index[1], end_timestamp=simple_ts_train.index[2]
    )
    pd.testing.assert_frame_equal(prediction.to_pandas(), expected_prediction.to_pandas())
