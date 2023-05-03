import pathlib
import tempfile
from typing import Sequence
from typing import Tuple

import pandas as pd
from lightning_fabric.utilities.seed import seed_everything
from optuna.samplers import RandomSampler

from etna.datasets import TSDataset
from etna.models.base import ModelType
from etna.pipeline import Pipeline
from etna.transforms import Transform


def get_loaded_model(model: ModelType) -> ModelType:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        model.save(path)
        loaded_model = model.load(path)
    return loaded_model


def assert_model_equals_loaded_original(
    model: ModelType, ts: TSDataset, transforms: Sequence[Transform], horizon: int
) -> Tuple[ModelType, ModelType]:

    pipeline_1 = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipeline_1.fit(ts)
    seed_everything(0)
    forecast_ts_1 = pipeline_1.forecast()

    loaded_model = get_loaded_model(pipeline_1.model)
    pipeline_1.model = loaded_model
    seed_everything(0)
    forecast_ts_2 = pipeline_1.forecast()

    pd.testing.assert_frame_equal(forecast_ts_1.to_pandas(), forecast_ts_2.to_pandas())

    return model, loaded_model


def assert_sampling_is_valid(model: ModelType, ts: TSDataset, seed: int = 0):
    grid = model.params_to_tune()
    # we need sampler to get a value from distribution
    sampler = RandomSampler(seed=seed)
    for name, distribution in grid.items():
        value = sampler.sample_independent(study=None, trial=None, param_name=name, param_distribution=distribution)
        new_model = model.set_params(**{name: value})
        new_model.fit(ts)


def assert_prediction_components_are_present(model, train, test, prediction_size=None):
    """Test that components are presented after in-sample and out-of-sample decomposition."""
    model.fit(train)

    predict_args = dict(ts=train, return_components=True)
    forecast_args = dict(ts=test, return_components=True)

    if prediction_size is not None:
        predict_args["prediction_size"] = prediction_size
        forecast_args["prediction_size"] = prediction_size

    forecast = model.predict(**predict_args)
    assert len(forecast.target_components_names) > 0

    forecast = model.forecast(**forecast_args)
    assert len(forecast.target_components_names) > 0
