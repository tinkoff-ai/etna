import optuna
import random
import copy

import hydra_slayer
import numpy as np
from etna.datasets import TSDataset, generate_ar_df
from etna.loggers import WandbLogger, tslogger
from etna.pipeline import Pipeline

from etna.metrics import MAE

SEED = 11
random.seed(SEED)
np.random.seed(SEED)


def init_logger(config):
    tslogger.loggers = []
    wblogger = WandbLogger(project="test-optuna", tags=["test", "optuna"], config=config)
    tslogger.add(wblogger)


def dataloader() -> TSDataset:
    df = generate_ar_df(periods=300, start_time="2021-01-02", n_segments=10)
    df.target = df.target + 100
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="1D")
    return ts


TS = dataloader()

preprocessing_map = dict(
    log_transform={"_target_": "etna.transforms.LogTransform", "in_column": "target"},
    add_const_transform={"_target_": "etna.transforms.AddConstTransform", "in_column": "target", "value": "100"},
)


def objective(trial: optuna.Trial):

    transforms = []

    preprocessing = trial.suggest_categorical("preprocessing", ["log_transform", "add_const_transform", "null"])

    if preprocessing == "log_transform":
        transforms.append(preprocessing_map[preprocessing])
    elif preprocessing == "add_const_transform":
        value = trial.suggest_int("add_const_transform_value", 10, 100, step=10)
        _copy = copy.copy(preprocessing_map[preprocessing])
        _copy["value"] = value
        transforms.append(_copy)
    else:
        pass

    lag = trial.suggest_int("model_lag", 14, 20, step=1)
    model_dict = {"_target_": "etna.models.NaiveModel", "lag": lag}

    init_logger({"transforms": transforms, "model_dict": model_dict})

    transforms = [hydra_slayer.get_from_params(**i) for i in transforms]

    pipeline = Pipeline(model=hydra_slayer.get_from_params(**model_dict), transforms=transforms, horizon=14)

    metrics, forecast, info = pipeline.backtest(ts=copy.deepcopy(TS), metrics=[MAE()])
    return metrics["MAE"].mean()


if __name__ == "__main__":

    study = optuna.create_study(
        storage="sqlite:///example.db",
        study_name="test",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        load_if_exists=True,
        direction="minimize",
    )

    study.optimize(objective, n_trials=200)
