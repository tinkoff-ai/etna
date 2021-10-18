from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.models import LinearPerSegmentModel
from etna.pipeline import AutoRegressivePipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform


def test_fit(example_tsds):
    """Test that AutoRegressivePipeline pipeline makes fit without failing."""
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform()]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=5, step=1)
    pipeline.fit(example_tsds)


def test_forecast_columns(example_tsds):
    """Test that AutoRegressivePipeline generates all the columns."""
    original_ts = deepcopy(example_tsds)
    horizon = 5

    # make predictions in AutoRegressivePipeline
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform(is_weekend=True)]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    # generate all columns
    original_ts.fit_transform(transforms)

    assert set(forecast_pipeline.columns) == set(original_ts.columns)


def test_forecast_one_step(example_tsds):
    """Test that AutoRegressivePipeline gets predictions one by one if step is equal to 1."""
    original_ts = deepcopy(example_tsds)
    horizon = 5

    # make predictions in AutoRegressivePipeline
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1])]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    # make predictions manually
    df = original_ts.to_pandas()
    original_ts.fit_transform(transforms)
    model = LinearPerSegmentModel()
    model.fit(original_ts)
    for i in range(horizon):
        cur_ts = TSDataset(df, freq=original_ts.freq)
        # these transform don't fit and we can fit_transform them at each step
        cur_ts.transform(transforms)
        cur_forecast_ts = cur_ts.make_future(1)
        cur_future_ts = model.forecast(cur_forecast_ts)
        to_add_df = cur_future_ts.to_pandas()
        df = pd.concat([df, to_add_df[df.columns]])

    forecast_manual = TSDataset(df.tail(horizon), freq=original_ts.freq)
    assert np.all(forecast_pipeline[:, :, "target"] == forecast_manual[:, :, "target"])


@pytest.mark.parametrize("horizon, step", ((1, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (20, 1), (20, 2), (20, 3)))
def test_forecast_multi_step(example_tsds, horizon, step):
    """Test that AutoRegressivePipeline gets correct number of predictions if step is more than 1."""
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[step])]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=step)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    assert forecast_pipeline.df.shape[0] == horizon


def test_forecast_with_fit_transforms(example_tsds):
    """Test that AutoRegressivePipeline can work with transforms that need fitting."""
    horizon = 5

    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), LinearTrendTransform(in_column="target")]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=horizon, step=1)
    pipeline.fit(example_tsds)
    pipeline.forecast()
