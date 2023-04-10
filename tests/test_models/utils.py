import pathlib
import tempfile
from functools import partial
from typing import Sequence
from typing import Tuple

import optuna
import pandas as pd
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
    import torch  # TODO: remove after fix at issue-802

    pipeline_1 = Pipeline(model=model, transforms=transforms, horizon=horizon)
    pipeline_1.fit(ts)
    torch.manual_seed(11)
    forecast_ts_1 = pipeline_1.forecast()

    loaded_model = get_loaded_model(pipeline_1.model)
    pipeline_1.model = loaded_model
    torch.manual_seed(11)
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


def assert_sample_params_makes_correct_suggest(
    model: ModelType,
    suggest_prefix: str,
    optuna_storage: optuna.storages.BaseStorage,
    n_trials: int = 3,
    seed: int = 0,
):
    study = optuna.create_study(
        storage=optuna_storage,
        study_name="example_name",
        sampler=optuna.samplers.RandomSampler(seed=seed),
        load_if_exists=True,
        direction="maximize",
    )

    def objective(trial, model):
        _ = model.sample_params(trial, suggest_prefix=suggest_prefix)
        trial_params = trial.params
        assert all(key.startswith(suggest_prefix) for key in trial_params.keys())
        return 1

    study.optimize(partial(objective, model=model), n_trials=n_trials)


def assert_sample_params_returns_correct_params(
    model: ModelType,
    ts: TSDataset,
    expected_num_params: int,
    optuna_storage: optuna.storages.BaseStorage,
    n_trials: int = 3,
    seed: int = 0,
):
    study = optuna.create_study(
        storage=optuna_storage,
        study_name="example_name",
        sampler=optuna.samplers.RandomSampler(seed=seed),
        load_if_exists=True,
        direction="maximize",
    )

    def objective(trial, model, ts):
        sampled_params = model.sample_params(trial)
        assert len(sampled_params) == expected_num_params
        new_model = model.set_params(**sampled_params)
        new_model.fit(ts)
        return 1

    study.optimize(partial(objective, model=model, ts=ts), n_trials=n_trials)
