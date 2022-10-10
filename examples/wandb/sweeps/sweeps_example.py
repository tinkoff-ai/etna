import random
from typing import Optional

import hydra
import hydra_slayer
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from pathlib import Path

from etna.datasets import TSDataset
from etna.loggers import WandbLogger
from etna.loggers import tslogger
from etna.pipeline import Pipeline

OmegaConf.register_new_resolver("range", lambda x, y: list(range(x, y)))
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)


FILE_PATH = Path(__file__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def init_logger(config: dict, project: str = "wandb-sweeps", tags: Optional[list] = ["test", "sweeps"]):
    tslogger.loggers = []
    wblogger = WandbLogger(project=project, tags=tags, config=config)
    tslogger.add(wblogger)


def dataloader(file_path: Path, freq: str) -> TSDataset:
    df = pd.read_csv(file_path)
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq=freq)
    return ts


@hydra.main(config_name="config.yaml")
def objective(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Load data
    ts = dataloader(file_path=cfg.dataset.file_path, freq=cfg.dataset.freq)

    # Init pipeline
    pipeline: Pipeline = hydra_slayer.get_from_params(**config["pipeline"])
    
    # Init backtest parameters like metrics and e.t.c.
    backtest_params = hydra_slayer.get_from_params(**config["backtest"])

    # Init WandB logger
    init_logger(pipeline.to_dict())

    # Run backtest
    _, _, _ = pipeline.backtest(ts, **backtest_params)


if __name__ == "__main__":
    objective()
