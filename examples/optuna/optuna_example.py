import random
from functools import partial
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import typer

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.loggers import WandbLogger
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import Sign
from etna.models import CatBoostModelMultiSegment
from etna.pipeline import Pipeline
from etna.transforms import LagTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import StandardScalerTransform

app = typer.Typer()


SEED = 11


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def init_logger(config):
    tslogger.loggers = []
    wblogger = WandbLogger(project="test-optuna", tags=["test", "optuna"], config=config)
    tslogger.add(wblogger)


def dataloader(file_path: Optional[str] = None, freq: str = "D") -> TSDataset:
    if file_path is not None:
        df = pd.read_csv(file_path)
    else:
        df = generate_ar_df(periods=300, start_time="2021-01-02", n_segments=10)
        df.target = df.target + 100

    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


def objective(trial: optuna.Trial, metric_name: str, ts: TSDataset, horizon: int):

    set_seed(SEED)

    pipeline = Pipeline(
        model=CatBoostModelMultiSegment(
            iterations=trial.suggest_int("iterations", 10, 100),
            depth=trial.suggest_int("depth", 1, 12),
        ),
        transforms=[
            StandardScalerTransform("target"),
            SegmentEncoderTransform(),
            LagTransform(in_column="target", lags=list(range(1, trial.suggest_int("lags", 2, 24)))),
        ],
        horizon=horizon,
    )

    init_logger(pipeline.to_dict())

    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE(), SMAPE(), Sign(), MSE()])
    return metrics[metric_name].mean()


@app.command()
def run_optuna(
    horizon: int = 14,
    metric_name: str = "MAE",
    storage: str = "sqlite:///optuna.db",
    study_name: Optional[str] = None,
    n_trials: int = 200,
    file_path: Optional[str] = None,
    direction: str = "minimize",
    freq: str = "D",
):
    """
    Run optuna optimization for CatBoostModelMultiSegment.
    """
    ts = dataloader(file_path, freq=freq)

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        load_if_exists=True,
        direction=direction,
    )

    study.optimize(partial(objective, metric_name=metric_name, ts=ts, horizon=horizon), n_trials=n_trials)


if __name__ == "__main__":
    typer.run(run_optuna)
