import pathlib
import tempfile
from copy import deepcopy
from typing import List
from typing import Tuple

import pandas as pd

from etna.datasets import TSDataset
from etna.pipeline.base import AbstractPipeline


def get_loaded_pipeline(pipeline: AbstractPipeline, ts: TSDataset) -> AbstractPipeline:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        pipeline.save(path)
        loaded_pipeline = pipeline.load(path, ts=ts)
    return loaded_pipeline


def select_segments_subset(ts: TSDataset, segments: List[str]) -> TSDataset:
    df = ts.raw_df.loc[:, pd.IndexSlice[segments, :]]
    df_exog = ts.df_exog
    if df_exog is not None:
        df_exog = df_exog.loc[:, pd.IndexSlice[segments, :]]
    known_future = ts.known_future
    freq = ts.freq
    return TSDataset(df=df, df_exog=df_exog, known_future=known_future, freq=freq)


def assert_pipeline_equals_loaded_original(
    pipeline: AbstractPipeline, ts: TSDataset
) -> Tuple[AbstractPipeline, AbstractPipeline]:
    import torch  # TODO: remove after fix at issue-802

    initial_ts = deepcopy(ts)

    pipeline.fit(ts)
    torch.manual_seed(11)
    forecast_ts_1 = pipeline.forecast()

    loaded_pipeline = get_loaded_pipeline(pipeline, ts=initial_ts)
    torch.manual_seed(11)
    forecast_ts_2 = loaded_pipeline.forecast()

    pd.testing.assert_frame_equal(forecast_ts_1.to_pandas(), forecast_ts_2.to_pandas())

    return pipeline, loaded_pipeline


def assert_pipeline_forecasts_with_given_ts(
    pipeline: AbstractPipeline, ts: TSDataset, segments_to_check: List[str]
) -> AbstractPipeline:
    import torch  # TODO: remove after fix at issue-802

    segments_to_check = list(set(segments_to_check))
    ts_selected = select_segments_subset(ts=deepcopy(ts), segments=segments_to_check)

    pipeline.fit(ts)
    torch.manual_seed(11)
    forecast_ts_1 = pipeline.forecast()
    forecast_df_1 = forecast_ts_1.to_pandas().loc[:, pd.IndexSlice[segments_to_check, :]]

    torch.manual_seed(11)
    forecast_ts_2 = pipeline.forecast(ts=ts_selected)
    forecast_df_2 = forecast_ts_2.to_pandas()

    pd.testing.assert_frame_equal(forecast_df_1, forecast_df_2)

    return pipeline
