import random
from functools import partial
from pathlib import Path
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

FILE_PATH = Path(__file__)

app = typer.Typer()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def init_logger(config: dict, project: str = "wandb-sweeps", tags: Optional[list] = ["test", "sweeps"]):
    tslogger.loggers = []
    wblogger = WandbLogger(project=project, tags=tags, config=config)
    tslogger.add(wblogger)


def dataloader(file_path: Path, freq: str = "D") -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


def objective(trial: optuna.Trial, metric_name: str, ts: TSDataset, horizon: int, lags: int, seed: int):
    """Optuna objective function."""

    # Set seed for reproducibility
    set_seed(seed)

    # Define model and features
    pipeline = Pipeline(
        model=CatBoostModelMultiSegment(
            iterations=trial.suggest_int("iterations", 10, 100),
            depth=trial.suggest_int("depth", 1, 12),
        ),
        transforms=[
            StandardScalerTransform("target"),
            SegmentEncoderTransform(),
            LagTransform(in_column="target", lags=list(range(horizon, horizon + trial.suggest_int("lags", 1, lags)))),
        ],
        horizon=horizon,
    )

    # Init WandB logger
    init_logger(pipeline.to_dict())

    # Start backtest
    metrics, _, _ = pipeline.backtest(ts=ts, metrics=[MAE(), SMAPE(), Sign(), MSE()])
    return metrics[metric_name].mean()


@app.command()
def run_optuna(
    horizon: int = 14,
    metric_name: str = "MAE",
    storage: str = "sqlite:///optuna.db",
    study_name: Optional[str] = None,
    n_trials: int = 200,
    file_path: Path = FILE_PATH.parents[1] / "data" / "example_dataset.csv",
    direction: str = "minimize",
    freq: str = "D",
    lags: int = 24,
    seed: int = 11,
):
    """
    Run optuna optimization for CatBoostModelMultiSegment.
    """
    # Load data
    ts = dataloader(file_path, freq=freq)

    # Create Optuna study
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        load_if_exists=True,
        direction=direction,
    )

    # Run Optuna optimization
    study.optimize(
        partial(objective, metric_name=metric_name, ts=ts, horizon=horizon, lags=lags, seed=seed), n_trials=n_trials
    )


if __name__ == "__main__":
    typer.run(run_optuna)
