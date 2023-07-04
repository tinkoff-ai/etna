from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from typing import cast

import hydra_slayer
import pandas as pd
import typer
from omegaconf import OmegaConf
from typing_extensions import Literal

from etna.commands.utils import estimate_max_n_folds
from etna.commands.utils import remove_params
from etna.datasets import TSDataset
from etna.pipeline import Pipeline

ADDITIONAL_BACKTEST_PARAMETERS = {"estimate_n_folds"}
ADDITIONAL_PIPELINE_PARAMETERS = {"context_size"}


def backtest(
    config_path: Path = typer.Argument(..., help="path to yaml config with desired pipeline"),
    backtest_config_path: Path = typer.Argument(..., help="path to backtest config file"),
    target_path: Path = typer.Argument(..., help="path to csv with data to forecast"),
    freq: str = typer.Argument(..., help="frequency of timestamp in files in pandas format"),
    output_path: Path = typer.Argument(..., help="where to save forecast"),
    exog_path: Optional[Path] = typer.Argument(default=None, help="path to csv with exog data"),
    known_future: Optional[List[str]] = typer.Argument(
        None,
        help="list of all known_future columns (regressor "
        "columns). If not specified then all exog_columns "
        "considered known_future.",
    ),
):
    """Command to run backtest with etna without coding.

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
    pipeline_configs = cast(Dict[str, Any], pipeline_configs)

    backtest_configs = OmegaConf.to_object(OmegaConf.load(backtest_config_path))

    df_timeseries = pd.read_csv(target_path, parse_dates=["timestamp"])

    df_timeseries = TSDataset.to_dataset(df_timeseries)

    df_exog = None
    k_f: Union[Literal["all"], Sequence[Any]] = ()
    if exog_path:
        df_exog = pd.read_csv(exog_path, parse_dates=["timestamp"])
        df_exog = TSDataset.to_dataset(df_exog)
        k_f = "all" if not known_future else known_future

    tsdataset = TSDataset(df=df_timeseries, freq=freq, df_exog=df_exog, known_future=k_f)

    pipeline_args = remove_params(params=pipeline_configs, to_remove=ADDITIONAL_PIPELINE_PARAMETERS)
    pipeline: Pipeline = hydra_slayer.get_from_params(**pipeline_args)
    backtest_configs_hydra_slayer: Dict[str, Any] = hydra_slayer.get_from_params(**backtest_configs)

    # estimate number of folds if parameters set
    if backtest_configs_hydra_slayer.get("estimate_n_folds", False):
        if "context_size" not in pipeline_configs:
            raise ValueError("Parameter `context_size` must be set if number of folds estimation enabled!")

        context_size = pipeline_configs["context_size"]

        max_n_folds = estimate_max_n_folds(
            ts=tsdataset,
            pipeline=pipeline,
            method_name="backtest",
            context_size=context_size,
            **backtest_configs_hydra_slayer,
        )

        n_folds = min(
            max_n_folds, backtest_configs_hydra_slayer.get("n_folds", 5)
        )  # use default value of folds if parameter not set

        backtest_configs_hydra_slayer["n_folds"] = n_folds

    backtest_call_args = remove_params(params=backtest_configs_hydra_slayer, to_remove=ADDITIONAL_BACKTEST_PARAMETERS)

    metrics, forecast, info = pipeline.backtest(ts=tsdataset, **backtest_call_args)

    (metrics.to_csv(output_path / "metrics.csv", index=False))
    (TSDataset.to_flatten(forecast).to_csv(output_path / "forecast.csv", index=False))
    (info.to_csv(output_path / "info.csv", index=False))


if __name__ == "__main__":
    typer.run(backtest)
