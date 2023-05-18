from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import hydra_slayer
import pandas as pd
import typer
from omegaconf import OmegaConf
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.models.utils import determine_num_steps
from etna.pipeline import Pipeline

ADDITIONAL_FORECAST_PARAMETERS = {"start_timestamp"}


def drop_additional_forecast_params(forecast_params: Dict[str, Any]) -> Dict[str, Any]:
    """Select `forecast` arguments from params."""
    return {k: v for k, v in forecast_params.items() if k not in ADDITIONAL_FORECAST_PARAMETERS}


def compute_horizon(horizon: int, forecast_params: Dict[str, Any], tsdataset: TSDataset) -> int:
    """Compute new pipeline horizon if `start_timestamp` presented in `forecast_params`."""
    if "start_timestamp" in forecast_params:
        freq = tsdataset.freq

        forecast_start_timestamp = pd.Timestamp(forecast_params["start_timestamp"], freq=freq)
        train_end_timestamp = tsdataset.index.max()

        if forecast_start_timestamp <= train_end_timestamp:
            raise ValueError("Parameter `start_timestamp` should greater than end of training dataset!")

        delta = determine_num_steps(
            start_timestamp=train_end_timestamp, end_timestamp=forecast_start_timestamp, freq=freq
        )

        horizon += delta - 1

    return horizon


def filter_forecast(forecast_ts: TSDataset, forecast_params: Dict[str, Any]) -> TSDataset:
    """Filter out forecasts before `start_timestamp` if `start_timestamp` presented in `forecast_params`.."""
    if "start_timestamp" in forecast_params:
        forecast_start_timestamp = pd.Timestamp(forecast_params["start_timestamp"], freq=forecast_ts.freq)
        forecast_ts.df = forecast_ts.df.loc[forecast_start_timestamp:, :]

    return forecast_ts


def forecast(
    config_path: Path = typer.Argument(..., help="path to yaml config with desired pipeline"),
    target_path: Path = typer.Argument(..., help="path to csv with data to forecast"),
    freq: str = typer.Argument(..., help="frequency of timestamp in files in pandas format"),
    output_path: Path = typer.Argument(..., help="where to save forecast"),
    exog_path: Optional[Path] = typer.Argument(None, help="path to csv with exog data"),
    forecast_config_path: Optional[Path] = typer.Argument(None, help="path to yaml config with forecast params"),
    raw_output: bool = typer.Argument(False, help="by default we return only forecast without features"),
    known_future: Optional[List[str]] = typer.Argument(
        None,
        help="list of all known_future columns (regressor "
        "columns). If not specified then all exog_columns "
        "considered known_future.",
    ),
):
    """Command to make forecast with etna without coding.

    Expected format of csv with target timeseries:

    \b
    =============  ===========  ==========
      timestamp      segment      target
    =============  ===========  ==========
    2020-01-01     segment_1         1
    2020-01-02     segment_1         2
    2020-01-03     segment_1         3
    2020-01-04     segment_1         4
    ...
    2020-01-10     segment_2        10
    2020-01-11     segment_2        20
    =============  ===========  ==========

    Expected format of csv with exogenous timeseries:

    \b
    =============  ===========  ===============  ===============
      timestamp      segment      regressor_1      regressor_2
    =============  ===========  ===============  ===============
    2020-01-01     segment_1          11               12
    2020-01-02     segment_1          22               13
    2020-01-03     segment_1          31               14
    2020-01-04     segment_1          42               15
    ...
    2020-02-10     segment_2         101               61
    2020-02-11     segment_2         205               54
    =============  ===========  ===============  ===============
    """
    pipeline_configs = OmegaConf.to_object(OmegaConf.load(config_path))
    if forecast_config_path:
        forecast_params_config = OmegaConf.to_object(OmegaConf.load(forecast_config_path))
    else:
        forecast_params_config = {}
    forecast_params: Dict[str, Any] = hydra_slayer.get_from_params(**forecast_params_config)

    df_timeseries = pd.read_csv(target_path, parse_dates=["timestamp"])

    df_timeseries = TSDataset.to_dataset(df_timeseries)

    df_exog = None
    k_f: Union[Literal["all"], Sequence[Any]] = ()
    if exog_path:
        df_exog = pd.read_csv(exog_path, parse_dates=["timestamp"])
        df_exog = TSDataset.to_dataset(df_exog)
        k_f = "all" if not known_future else known_future

    tsdataset = TSDataset(df=df_timeseries, freq=freq, df_exog=df_exog, known_future=k_f)

    horizon: int = pipeline_configs["horizon"]  # type: ignore
    horizon = compute_horizon(horizon=horizon, forecast_params=forecast_params, tsdataset=tsdataset)
    pipeline_configs["horizon"] = horizon  # type: ignore

    forecast_args = drop_additional_forecast_params(forecast_params)

    pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline_configs)
    pipeline.fit(tsdataset)
    forecast = pipeline.forecast(**forecast_args)

    forecast = filter_forecast(forecast_ts=forecast, forecast_params=forecast_params)

    flatten = forecast.to_pandas(flatten=True)
    if raw_output:
        (flatten.to_csv(output_path, index=False))
    else:
        quantile_columns = [column for column in flatten.columns if column.startswith("target_0.")]
        (flatten[["timestamp", "segment", "target"] + quantile_columns].to_csv(output_path, index=False))


if __name__ == "__main__":
    typer.run(forecast)
