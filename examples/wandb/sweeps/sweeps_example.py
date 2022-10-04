import random
from typing import Optional

import hydra
import hydra_slayer
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.loggers import WandbLogger
from etna.loggers import tslogger
from etna.pipeline import Pipeline

OmegaConf.register_new_resolver("range", lambda x, y: list(range(x, y)))

SEED = 11


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def init_logger(config):
    tslogger.loggers = []
    wblogger = WandbLogger(project="test-wandb-sweeps", tags=["test", "sweeps"], config=config)
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


@hydra.main(config_name="config.yaml")
def objective(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)

    pipeline = config["pipeline"]
    backtest = config["backtest"]

    ts = dataloader(file_path=cfg.dataset.file_path, freq=cfg.dataset.freq)

    pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline)
    backtest_configs = hydra_slayer.get_from_params(**backtest)

    init_logger(pipeline.to_dict())

    metrics, _, _ = pipeline.backtest(ts, **backtest_configs)
    return metrics["MAE"].mean()


if __name__ == "__main__":
    objective()
