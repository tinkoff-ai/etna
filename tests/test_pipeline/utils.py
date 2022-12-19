import pathlib
import tempfile
from copy import deepcopy
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
