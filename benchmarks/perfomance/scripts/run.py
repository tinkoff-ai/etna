import random
from pathlib import Path
from typing import Any, Dict

import hydra_slayer
import numpy as np
from omegaconf import OmegaConf

from etna.datasets import TSDataset, generate_ar_df
from etna.pipeline import Pipeline

FILE_PATH = Path(__file__).parent

config_path = Path.cwd() / ".hydra" / "config.yaml"

config = OmegaConf.load(config_path)
config_dict = OmegaConf.to_object(OmegaConf.load(config_path))

random.seed(config.seed)
np.random.seed(config.seed)


pipeline_configs = config_dict["pipeline"]
backtest_configs = config_dict["backtest"]

pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline_configs)
backtest_configs_hydra_slayer: Dict[str, Any] = hydra_slayer.get_from_params(**backtest_configs)

df_timeseries = generate_ar_df(
    periods=config.dataset.periods,
    start_time="2021-06-01",
    n_segments=config.dataset.n_segments,
    freq=config.dataset.freq,
)


df_timeseries = TSDataset.to_dataset(df_timeseries)

df_exog = None

tsdataset = TSDataset(df=df_timeseries, freq=config.dataset.freq, df_exog=df_exog)
metrics, forecast, info = pipeline.backtest(ts=tsdataset, **backtest_configs_hydra_slayer)
