"""
Example of using WandB with ETNA library.
Current script could be used for sweeps and simple validation runs.
"""

import argparse
import random
from typing import Any, Dict

import hydra_slayer
import numpy as np
from etna.datasets import TSDataset, generate_ar_df
from etna.loggers import WandbLogger, tslogger
from etna.pipeline import Pipeline
from omegaconf import OmegaConf

SEED = 11
random.seed(SEED)
np.random.seed(SEED)

# Default config loading
config = OmegaConf.load("pipeline.yaml")


# Define arguments for WandB sweep parameters
args = argparse.ArgumentParser()
args.add_argument("--iterations", type=int)
args.add_argument("--learning-rate", type=float)
for key, value in vars(args.parse_args()).items():
    if value:
        config[key] = value

# Config for Pipeline and backtesting pipeline
config = OmegaConf.to_container(config, resolve=True)
pipeline = config["pipeline"]
backtest = config["backtest"]


# Define WandbLogger and passing it to global library logger
# It will not log child processes in case of `spawn` (OSX or Windows)
wblogger = WandbLogger(project="test-run", config=pipeline)
tslogger.add(wblogger)


def dataloader() -> TSDataset:
    df = generate_ar_df(periods=300, start_time="2021-01-02", n_segments=10)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="1D")
    return ts


if __name__ == "__main__":

    ts = dataloader()

    pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline)

    backtest_configs: Dict[str, Any] = hydra_slayer.get_from_params(**backtest)

    metrics, forecast, info = pipeline.backtest(ts=ts, **backtest_configs)