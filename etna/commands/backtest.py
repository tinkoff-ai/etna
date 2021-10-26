from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import hydra_slayer
import pandas as pd
import typer
import yaml

from etna.datasets import TSDataset
from etna.pipeline import Pipeline


def backtest(
    config_path: Path = typer.Argument(..., help="path to yaml config with desired pipeline"),
    backtest_config_path: Path = typer.Argument(..., help="path to backtest config file"),
    target_path: Path = typer.Argument(..., help="path to csv with data to forecast"),
    freq: str = typer.Argument(..., help="frequency of timestamp in files in pandas format"),
    output_path: Path = typer.Argument(..., help="where to save forecast"),
    exog_path: Optional[Path] = typer.Argument(default=None, help="path to csv with exog data"),
):
    """Command to make forecast with etna without coding.

    Expected format of csv with target timeseries:

    \b
    | timestamp           | segment   |   target |
    |:--------------------|:----------|---------:|
    | 2019-01-01 00:00:00 | segment_a |      170 |
    | 2019-01-02 00:00:00 | segment_a |      243 |
    | 2019-01-03 00:00:00 | segment_a |      267 |
    | 2019-01-04 00:00:00 | segment_a |      287 |
    | 2019-01-05 00:00:00 | segment_a |      279 |

    Expected format of csv with exogenous timeseries:

    \b
    | timestamp           |   regressor_1 |   regressor_2 | segment   |
    |:--------------------|--------------:|--------------:|:----------|
    | 2019-01-01 00:00:00 |             0 |             5 | segment_a |
    | 2019-01-02 00:00:00 |             1 |             6 | segment_a |
    | 2019-01-03 00:00:00 |             2 |             7 | segment_a |
    | 2019-01-04 00:00:00 |             3 |             8 | segment_a |
    | 2019-01-05 00:00:00 |             4 |             9 | segment_a |
    """
    with open(config_path, "r") as f:
        pipeline = yaml.safe_load(f)

    with open(backtest_config_path, "r") as f:
        backtest_configs = yaml.safe_load(f)

    df_timeseries = pd.read_csv(target_path, parse_dates=["timestamp"])

    df_timeseries = TSDataset.to_dataset(df_timeseries)

    df_exog = None
    if exog_path:
        df_exog = pd.read_csv(exog_path, parse_dates=["timestamp"])
        df_exog = TSDataset.to_dataset(df_exog)

    tsdataset = TSDataset(df=df_timeseries, freq=freq, df_exog=df_exog)

    pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline)
    backtest_configs: Dict[str, Any] = hydra_slayer.get_from_params(**backtest_configs)

    metrics, forecast, info = pipeline.backtest(ts=tsdataset, **backtest_configs)

    (metrics.to_csv(output_path / "metrics.csv", index=False))
    (forecast.to_csv(output_path / "forecast.csv", index=False))
    (info.to_csv(output_path / "info.csv", index=False))


if __name__ == "__main__":
    typer.run(backtest)
