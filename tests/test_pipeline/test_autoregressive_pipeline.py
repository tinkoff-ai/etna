from copy import deepcopy
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode
from etna.models import LinearPerSegmentModel
from etna.pipeline import AutoRegressivePipeline
from etna.transforms import LagTransform
from etna.transforms import DateFlagsTransform


def test_fit(example_tsds):
    """Test that AutoRegressivePipeline pipeline makes fit without failing."""
    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform()]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=5, step=1)
    pipeline.fit(example_tsds)


def test_forecast(example_tsds):
    """Test that AutoRegressivePipeline gets predictions one by one."""
    original_ts = deepcopy(example_tsds)

    model = LinearPerSegmentModel()
    transforms = [LagTransform(in_column="target", lags=[1]), DateFlagsTransform()]
    pipeline = AutoRegressivePipeline(model=model, transforms=transforms, horizon=5, step=1)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast()

    # TODO: write meaningful test!!!
    original_ts.fit_transform(transforms)
    model.fit(original_ts)
    future = original_ts.make_future(5)
    forecast_manual = model.forecast(future)

    assert np.all(forecast_pipeline.df.values == forecast_manual.df.values)
