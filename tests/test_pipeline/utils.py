import pathlib
import tempfile
from copy import deepcopy
from typing import Tuple

import pandas as pd
from lightning_fabric.utilities.seed import seed_everything

from etna.datasets import TSDataset
from etna.pipeline.base import AbstractPipeline


def get_loaded_pipeline(pipeline: AbstractPipeline, ts: TSDataset = None) -> AbstractPipeline:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        pipeline.save(path)
        if ts is None:
            loaded_pipeline = pipeline.load(path)
        else:
            loaded_pipeline = pipeline.load(path, ts=ts)
    return loaded_pipeline


def assert_pipeline_equals_loaded_original(
    pipeline: AbstractPipeline, ts: TSDataset, load_ts: bool = True
) -> Tuple[AbstractPipeline, AbstractPipeline]:

    initial_ts = deepcopy(ts)

    pipeline.fit(ts)
    seed_everything(0)
    forecast_ts_1 = pipeline.forecast()

    if load_ts:
        loaded_pipeline = get_loaded_pipeline(pipeline, ts=initial_ts)
        seed_everything(0)
        forecast_ts_2 = loaded_pipeline.forecast()
    else:
        loaded_pipeline = get_loaded_pipeline(pipeline)
        seed_everything(0)
        forecast_ts_2 = loaded_pipeline.forecast(ts=initial_ts)

    pd.testing.assert_frame_equal(forecast_ts_1.to_pandas(), forecast_ts_2.to_pandas())

    return pipeline, loaded_pipeline


def assert_pipeline_forecasts_given_ts(pipeline: AbstractPipeline, ts: TSDataset, horizon: int) -> AbstractPipeline:
    fit_ts = deepcopy(ts)
    fit_ts.df = fit_ts.df.iloc[:-horizon]
    to_forecast_ts = deepcopy(ts)

    pipeline.fit(ts=fit_ts)
    forecast_ts = pipeline.forecast(ts=to_forecast_ts)
    forecast_df = forecast_ts.to_pandas(flatten=True)

    if ts.has_hierarchy():
        expected_segments = ts.hierarchical_structure.get_level_segments(forecast_ts.current_df_level)
    else:
        expected_segments = to_forecast_ts.segments
    assert forecast_ts.segments == expected_segments
    expected_index = pd.date_range(
        start=to_forecast_ts.index[-1], periods=horizon + 1, freq=to_forecast_ts.freq, name="timestamp"
    )[1:]
    pd.testing.assert_index_equal(forecast_ts.index, expected_index)
    assert not forecast_df["target"].isna().any()

    return pipeline


def assert_pipeline_forecasts_given_ts_with_prediction_intervals(
    pipeline: AbstractPipeline, ts: TSDataset, horizon: int, **forecast_params
) -> AbstractPipeline:
    fit_ts = deepcopy(ts)
    fit_ts.df = fit_ts.df.iloc[:-horizon]
    to_forecast_ts = deepcopy(ts)

    pipeline.fit(fit_ts)
    forecast_ts = pipeline.forecast(
        ts=to_forecast_ts, prediction_interval=True, quantiles=[0.025, 0.975], **forecast_params
    )
    forecast_df = forecast_ts.to_pandas(flatten=True)

    if ts.has_hierarchy():
        expected_segments = ts.hierarchical_structure.get_level_segments(forecast_ts.current_df_level)
    else:
        expected_segments = to_forecast_ts.segments
    assert forecast_ts.segments == expected_segments
    expected_index = pd.date_range(
        start=to_forecast_ts.index[-1], periods=horizon + 1, freq=to_forecast_ts.freq, name="timestamp"
    )[1:]
    pd.testing.assert_index_equal(forecast_ts.index, expected_index)
    assert not forecast_df["target"].isna().any()
    assert not forecast_df["target_0.025"].isna().any()
    assert not forecast_df["target_0.975"].isna().any()

    return pipeline
