import pathlib
import random

import numpy as np
import pandas as pd
import torch
import typer

from etna.datasets.tsdataset import TSDataset
from etna.loggers import LocalFileLogger, WandbLogger, tslogger
from etna.metrics import MAE, MSE, SMAPE, Sign
from etna.models.nn import RNNModel
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, StandardScalerTransform


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_backtest(
    horizon: int = 7,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    seed: int = 11,
    dataset_path: pathlib.Path = pathlib.Path("data/example_dataset.csv"),
    experiments_folder: pathlib.Path = pathlib.Path("experiments"),
    dataset_freq: str = "D",
    num_lags: int = 10,
):
    parameters = dict(locals())
    parameters["dataset_path"] = str(dataset_path)
    parameters["experiments_folder"] = str(experiments_folder)

    set_seed(seed)

    original_df = pd.read_csv(dataset_path)
    df = TSDataset.to_dataset(original_df)
    ts = TSDataset(df, freq=dataset_freq)

    model_rnn = RNNModel(
        decoder_length=horizon,
        encoder_length=2 * horizon,
        input_size=num_lags + 1,
        trainer_params={"max_epochs": n_epochs},
        lr=lr,
        train_batch_size=batch_size,
    )

    transform_lag = LagTransform(
        in_column="target",
        lags=[horizon + i for i in range(num_lags)],
        out_column="target_lag",
    )
    pipeline_rnn = Pipeline(
        model=model_rnn,
        horizon=horizon,
        transforms=[StandardScalerTransform(in_column="target"), transform_lag],
    )

    tslogger.add(LocalFileLogger(config=parameters, experiments_folder=experiments_folder))
    tslogger.add(
        WandbLogger(
            project="test-demo",
            config=parameters,
        )
    )

    metrics = [SMAPE(), MSE(), MAE(), Sign()]
    metrics_rnn, forecast_rnn, fold_info_rnn = pipeline_rnn.backtest(ts, metrics=metrics, n_folds=3, n_jobs=1)


if __name__ == "__main__":
    typer.run(train_backtest)
