import random
import copy
import hydra
from omegaconf import DictConfig, OmegaConf

import hydra_slayer
import numpy as np
from etna.datasets import TSDataset, generate_ar_df
from etna.loggers import WandbLogger, tslogger

SEED = 11
random.seed(SEED)
np.random.seed(SEED)


def init_logger(config):
    tslogger.loggers = []
    wblogger = WandbLogger(project="test-hydra--sweeps", tags=["test", "hydra"], config=config)
    tslogger.add(wblogger)


def dataloader() -> TSDataset:
    df = generate_ar_df(periods=300, start_time="2021-01-02", n_segments=10)
    df.target = df.target + 100
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df=df, freq="1D")
    return ts


TS = dataloader()


@hydra.main(config_name="pipeline.yaml")
def objective(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    pipeline = config["pipeline"]
    backtest = config["backtest"]

    if pipeline["transforms"][0] == "None":
        pipeline["transforms"] = []

    init_logger(copy.copy(pipeline))

    pipeline = hydra_slayer.get_from_params(**pipeline)
    backtest_configs = hydra_slayer.get_from_params(**backtest)

    metrics, forecast, info = pipeline.backtest(copy.deepcopy(TS), **backtest_configs)
    return metrics["MAE"].mean()


if __name__ == "__main__":
    objective()
