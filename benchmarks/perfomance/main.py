from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import hydra_slayer
import pandas as pd
import typer
from omegaconf import OmegaConf

from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.datasets import generate_ar_df

from scalene import scalene_profiler

import time 



config_path = "configs/pipeline.yaml"
backtest_config_path = "configs/backtest.yaml"
pipeline_configs = OmegaConf.to_object(OmegaConf.load(config_path))
backtest_configs = OmegaConf.to_object(OmegaConf.load(backtest_config_path))

pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline_configs)
backtest_configs_hydra_slayer: Dict[str, Any] = hydra_slayer.get_from_params(**backtest_configs)

#df_timeseries = pd.read_csv(target_path, parse_dates=["timestamp"])

start_time = time.monotonic()
df_timeseries = generate_ar_df(periods=100, start_time="2021-06-01", n_segments=100)


df_timeseries = TSDataset.to_dataset(df_timeseries)


df_exog = None


tsdataset = TSDataset(df=df_timeseries, freq="1D", df_exog=df_exog)
metrics, forecast, info = pipeline.backtest(ts=tsdataset, **backtest_configs_hydra_slayer)
print((time.monotonic() - start_time))

scalene_profiler.start()
start_time = time.monotonic()

df_timeseries = generate_ar_df(periods=100, start_time="2021-06-01", n_segments=100)


df_timeseries = TSDataset.to_dataset(df_timeseries)


df_exog = None


tsdataset = TSDataset(df=df_timeseries, freq="1D", df_exog=df_exog)
metrics, forecast, info = pipeline.backtest(ts=tsdataset, **backtest_configs_hydra_slayer)
print((time.monotonic() - start_time))

scalene_profiler.stop()


# python3 -m scalene --off  --profile-all --profile-only etna-ts/etna,etna/etna --html --outfile=ll.html main.py configs/pipeline.yaml configs/backtest.yaml 