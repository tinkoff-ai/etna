from pathlib import Path
from typing import Optional

import hydra_slayer
import pandas as pd
import typer
import yaml

from etna.datasets import TSDataset
from etna.pipeline import Pipeline
from etna.metrics import MAE, MSE, MAPE, SMAPE


def make_flatten(df: pd.DataFrame) -> pd.DataFrame:
    aggregator_list = []
    category = []
    segments = set(df.columns.get_level_values(0))
    for segment in segments:
        if df[segment].select_dtypes(include=["category"]).columns.to_list():
            category.extend(df[segment].select_dtypes(include=["category"]).columns.to_list())
        aggregator_list.append(df[segment].copy())
        aggregator_list[-1]["segment"] = segment
    df = pd.concat(aggregator_list)
    df = df.reset_index()
    category = list(set(category))
    df[category] = df[category].astype("category")
    return df


def backtest(
    config_path: Path = typer.Argument(..., help="path to yaml config with desired pipeline"),
    target_path: Path = typer.Argument(..., help="path to csv with data to forecast"),
    freq: str = typer.Argument(..., help="frequency of timestamp in files in pandas format"),
    output_path: Path = typer.Argument(..., help="where to save forecast"),
    exog_path: Optional[Path] = typer.Argument(None, help="path to csv with exog data"),
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

    df_timeseries = pd.read_csv(target_path, parse_dates=["timestamp"], index_col=0)

    df_timeseries = TSDataset.to_dataset(df_timeseries)

    df_exog = None
    if exog_path:
        df_exog = pd.read_csv(exog_path, parse_dates=["timestamp"])
        df_exog = TSDataset.to_dataset(df_exog)

    tsdataset = TSDataset(df=df_timeseries, freq=freq, df_exog=df_exog)

    pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline)
    # pipeline.fit(tsdataset)
    metrics, forecast, info = pipeline.backtest(
        ts=tsdataset,
        n_folds=3,
        n_jobs=3,
        metrics=[MAE(), MAPE(), SMAPE(), MSE()]
    )

    (metrics.to_csv(output_path / "metrics.csv", index=False))
    (make_flatten(forecast).to_csv(output_path / "forecast.csv", index=False))
    (info.to_csv(output_path / "info.csv", index=False))


if __name__ == "__main__":
    typer.run(backtest)
