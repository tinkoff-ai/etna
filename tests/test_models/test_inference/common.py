import functools

import numpy as np
import pytest
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models import ContextRequiredModelType


def to_be_fixed(raises, match=None):
    def to_be_fixed_concrete(func):
        @functools.wraps(func)
        def wrapped_test(*args, **kwargs):
            with pytest.raises(raises, match=match):
                return func(*args, **kwargs)

        return wrapped_test

    return to_be_fixed_concrete


def _test_prediction_in_sample_full(ts, model, transforms, method_name):
    df = ts.to_pandas()
    method = getattr(model, method_name)

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)

    if isinstance(model, get_args(ContextRequiredModelType)):
        prediction_size = len(forecast_ts.index)
        method(forecast_ts, prediction_size=prediction_size)
    else:
        method(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())


def _test_prediction_in_sample_suffix(ts, model, transforms, method_name, num_skip_points):
    df = ts.to_pandas()
    method = getattr(model, method_name)

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)

    if isinstance(model, get_args(ContextRequiredModelType)):
        prediction_size = len(forecast_ts.index) - num_skip_points
        method(forecast_ts, prediction_size=prediction_size)
    else:
        forecast_ts.df = forecast_ts.df.iloc[num_skip_points:]
        method(forecast_ts)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())
