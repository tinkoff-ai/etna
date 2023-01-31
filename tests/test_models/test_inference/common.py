import functools
from typing import List

import numpy as np
import pandas as pd
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


def make_prediction(model, ts, prediction_size, method_name) -> TSDataset:
    method = getattr(model, method_name)
    if isinstance(model, get_args(ContextRequiredModelType)):
        ts = method(ts, prediction_size=prediction_size)
    else:
        ts = method(ts)
    return ts


def select_segments_subset(ts: TSDataset, segments: List[str]) -> TSDataset:
    df = ts.raw_df.loc[:, pd.IndexSlice[segments, :]]
    df_exog = ts.df_exog
    if df_exog is not None:
        df_exog = df_exog.loc[:, pd.IndexSlice[segments, :]]
    known_future = ts.known_future
    freq = ts.freq
    return TSDataset(df=df, df_exog=df_exog, known_future=known_future, freq=freq)


def _test_prediction_in_sample_full(ts, model, transforms, method_name):
    df = ts.to_pandas()

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)
    prediction_size = len(forecast_ts.index)
    forecast_ts = make_prediction(model=model, ts=forecast_ts, prediction_size=prediction_size, method_name=method_name)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())
    original_target = TSDataset.to_flatten(df)["target"]
    assert not forecast_df["target"].equals(original_target)


def _test_prediction_in_sample_suffix(ts, model, transforms, method_name, num_skip_points):
    df = ts.to_pandas()

    # fitting
    ts.fit_transform(transforms)
    model.fit(ts)

    # forecasting
    forecast_ts = TSDataset(df, freq="D")
    forecast_ts.transform(ts.transforms)
    prediction_size = len(forecast_ts.index) - num_skip_points
    forecast_ts.df = forecast_ts.df.iloc[(num_skip_points - model.context_size) :]
    forecast_ts = make_prediction(model=model, ts=forecast_ts, prediction_size=prediction_size, method_name=method_name)

    # checking
    forecast_df = forecast_ts.to_pandas(flatten=True)
    assert not np.any(forecast_df["target"].isna())
    original_target = TSDataset.to_flatten(df.iloc[(num_skip_points - model.context_size) :])["target"]
    assert not forecast_df["target"].equals(original_target)
