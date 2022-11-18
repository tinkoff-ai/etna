import pathlib
import tempfile
from copy import deepcopy
from typing import Sequence
from typing import Tuple

import pandas as pd

from etna.datasets import TSDataset
from etna.models.base import ModelType
from etna.pipeline import Pipeline
from etna.transforms import Transform


def get_loaded_model(model: ModelType) -> ModelType:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        model.save(path)
        loaded_model = deepcopy(model).load(path)
    return loaded_model


def assert_model_equals_loaded_original(
    model: ModelType, ts: TSDataset, transforms: Sequence[Transform], horizon: int
) -> Tuple[ModelType, ModelType]:
    pipeline_1 = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipeline_1.fit(ts)
    loaded_model = get_loaded_model(pipeline_1.model)
    pipeline_2 = deepcopy(pipeline_1)
    pipeline_2.model = loaded_model

    forecast_ts_1 = pipeline_1.forecast()
    forecast_ts_2 = pipeline_2.forecast()

    pd.testing.assert_frame_equal(forecast_ts_1.to_pandas(), forecast_ts_2.to_pandas())

    return model, loaded_model
