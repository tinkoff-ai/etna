import base64
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger

from etna.analysis import plot_backtest
from etna.loggers.base import BaseLogger

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def percentile(n: int):
    """Percentile for pandas agg."""

    def percentile_(x):
        return np.percentile(x.values, n)

    percentile_.__name__ = "percentile_%s" % n
    return percentile_


class WBLogger(BaseLogger):
    """Weights&Biases logger."""

    name = "wblogger"

    def __init__(
        self,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        job_type: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        plot: bool = True,
        table: bool = True,
        name_prefix: str = "",
    ):
        super().__init__()
        self.name = (
            name_prefix + base64.urlsafe_b64encode(uuid4().bytes).decode("utf8").rstrip("=\n")[:8]
            if name is None
            else name
        )
        self.project = project
        self.entity = entity
        self.group = group
        self.config = None
        self._experiment = None
        self._pl_logger = None
        self.job_type = job_type
        self.tags = tags
        self.plot = plot
        self.table = table
        self.name_prefix = name_prefix

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """
        Log any event.

        e.g. "Fitted segment segment_name" to stderr output.

        Parameters
        ----------
        msg:
            Message or dict to log
        kwargs:
            Parameters for changing additional info in log message
        """
        pass

    def log_backtest_metrics(
        self, ts: "TSDataset", metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

        Parameters
        ----------
        metrics_df: pd.DataFrame
            Dataframe produced with TimeSeriesCrossValidation.get_metrics(aggregate_metrics=False)
        forecast_df: pd.DataFrame
            Forecast from backtest
        fold_info_df: pd.DataFrame
            Fold information from backtest
        """
        if self.table:
            self.experiment.summary["metrics"] = wandb.Table(data=metrics_df)
            self.experiment.summary["forecast"] = wandb.Table(data=forecast_df)
            self.experiment.summary["fold_info"] = wandb.Table(data=fold_info_df)

        # TODO: make it faster
        if self.plot:
            plot_backtest(forecast_df, ts, history_len=100)
            self.experiment.log({"backtest": plt})

        metrics_dict = (
            metrics_df.groupby("segment")
            .mean()
            .reset_index()
            .drop(["segment", "fold_number"], axis=1)
            .apply(["median", "mean", "std", percentile(5), percentile(25), percentile(75), percentile(95)])
            .to_dict()
        )
        for metrics_key, values in metrics_dict.items():
            for statistics_key, value in values.items():
                self.experiment.summary[f"{metrics_key}_{statistics_key}"] = value

    def log_backtest_run(self, metrics: pd.DataFrame, forecast: pd.DataFrame, test: pd.DataFrame):
        """
        Backtest metrics from one fold to logger.

        Parameters
        ----------
        metrics: pd.DataFrame
            Dataframe with metrics from backtest fold
        forecast: pd.DataFrame
            Dataframe with forecast
        test: pd.DataFrame
            Dataframe with ground trouth
        """
        columns_name = list(metrics.columns)
        metrics.reset_index(inplace=True)
        metrics.columns = ["segment"] + columns_name
        if self.table:
            self.experiment.summary["metrics"] = wandb.Table(data=metrics)
            self.experiment.summary["forecast"] = wandb.Table(data=forecast)
            self.experiment.summary["test"] = wandb.Table(data=test)

        # too slow
        # plot_forecast(forecast, test)
        # self.experiment.log({"train vs test": plt})

        metrics_dict = (
            metrics.drop(["segment"], axis=1)
            .apply(["median", "mean", "std", percentile(5), percentile(25), percentile(75), percentile(95)])
            .to_dict()
        )
        for metrics_key, values in metrics_dict.items():
            for statistics_key, value in values.items():
                self.experiment.summary[f"{metrics_key}_{statistics_key}"] = value

    def set_config(self, forecaster):
        """Pass forecaster config to loggers."""
        self.config = forecaster.to_json()

    def start_experiment(self, job_type: str = None, group: str = None, *args, **kwargs):
        """Start experiment(logger post init or reinit next experiment with the same name)."""
        self.job_type = job_type
        self.group = group
        self.reinit_experiment()
        self._pl_logger = WandbLogger(experiment=self.experiment)

    def reinit_experiment(self):
        """Reinit experiment."""
        self._experiment = wandb.init(
            name=self.name,
            project=self.project,
            group=self.group,
            config=self.config,
            reinit=True,
            tags=self.tags,
            job_type=self.job_type,
            settings=wandb.Settings(start_method="thread"),
        )

    def finish_experiment(self):
        """Finish experiment."""
        self._experiment.finish()

    @property
    def pl_logger(self):
        """Pytorch lightning loggers."""
        self._pl_logger = WandbLogger(experiment=self.experiment, log_model=True)
        return self._pl_logger

    @property
    def experiment(self):
        """Init experiment."""
        if self._experiment is None:
            self.reinit_experiment()
            self._pl_logger = WandbLogger(experiment=self.experiment)
        return self._experiment
