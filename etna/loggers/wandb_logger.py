import base64
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import uuid4

import numpy as np
import pandas as pd

from etna import SETTINGS
from etna.loggers.base import BaseLogger

if TYPE_CHECKING:
    from pytorch_lightning.loggers import WandbLogger as PLWandbLogger

    from etna.datasets import TSDataset

if SETTINGS.wandb_required:
    import wandb


def percentile(n: int):
    """Percentile for pandas agg."""

    def percentile_(x):
        return np.percentile(x.values, n)

    percentile_.__name__ = "percentile_%s" % n
    return percentile_


class WandbLogger(BaseLogger):
    """Weights&Biases logger."""

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
        config: Optional[Union[Dict, str, None]] = None,
        log_model: bool = False,
    ):
        """
        Create instance of WandbLogger.

        Parameters
        ----------
        name:
            Wandb run name.
        entity:
            An entity is a username or team name where you're sending runs.
        project:
            The name of the project where you're sending the new run
        job_type:
            Specify the type of run, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual runs into a larger experiment.
        tags:
            A list of strings, which will populate the list of tags on this run in the UI.
        plot:
            Indicator for making and sending plots.
        table:
            Indicator for making and sending tables.
        name_prefix:
            Prefix for the name field.
        config:
            This sets `wandb.config`, a dictionary-like object for saving inputs to your job,
            like hyperparameters for a model or settings for a data preprocessing job.
        """
        super().__init__()
        self.name = (
            name_prefix + base64.urlsafe_b64encode(uuid4().bytes).decode("utf8").rstrip("=\n")[:8]
            if name is None
            else name
        )
        self.project = project
        self.entity = entity
        self.group = group
        self.config = config
        self._experiment = None
        self._pl_logger: Optional["PLWandbLogger"] = None
        self.job_type = job_type
        self.tags = tags
        self.plot = plot
        self.table = table
        self.name_prefix = name_prefix
        self.log_model = log_model

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

        Notes
        -----
        We log nothing via current method in wandb case.
        Currently you could call ``wandb.log`` by hand if you need this.
        """
        pass

    def log_backtest_metrics(
        self, ts: "TSDataset", metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

        Parameters
        ----------
        metrics_df:
            Dataframe produced with Pipeline._get_backtest_metrics() or TimeSeriesCrossValidation.get_metrics()
        forecast_df:
            Forecast from backtest
        fold_info_df:
            Fold information from backtest
        """
        from etna.analysis import plot_backtest_interactive
        from etna.datasets import TSDataset

        if self.table:
            self.experiment.summary["metrics"] = wandb.Table(data=metrics_df)
            self.experiment.summary["forecast"] = wandb.Table(data=TSDataset.to_flatten(forecast_df))
            self.experiment.summary["fold_info"] = wandb.Table(data=fold_info_df)

        if self.plot:
            fig = plot_backtest_interactive(forecast_df, ts, history_len=100)
            self.experiment.log({"backtest": fig})

        # case for aggregate_metrics=False
        if "fold_number" in metrics_df.columns:
            metrics_dict = (
                metrics_df.groupby("segment")
                .mean()
                .reset_index()
                .drop(["segment", "fold_number"], axis=1)
                .apply(["median", "mean", "std", percentile(5), percentile(25), percentile(75), percentile(95)])
                .to_dict()
            )
        # case for aggregate_metrics=True
        else:
            metrics_dict = (
                metrics_df.drop(["segment"], axis=1)
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
        metrics:
            Dataframe with metrics from backtest fold
        forecast:
            Dataframe with forecast
        test:
            Dataframe with ground truth
        """
        from etna.datasets import TSDataset

        columns_name = list(metrics.columns)
        metrics.reset_index(inplace=True)
        metrics.columns = ["segment"] + columns_name
        if self.table:
            self.experiment.summary["metrics"] = wandb.Table(data=metrics)
            self.experiment.summary["forecast"] = wandb.Table(data=TSDataset.to_flatten(forecast))
            self.experiment.summary["test"] = wandb.Table(data=TSDataset.to_flatten(test))

        metrics_dict = (
            metrics.drop(["segment"], axis=1)
            .apply(["median", "mean", "std", percentile(5), percentile(25), percentile(75), percentile(95)])
            .to_dict()
        )
        for metrics_key, values in metrics_dict.items():
            for statistics_key, value in values.items():
                self.experiment.summary[f"{metrics_key}_{statistics_key}"] = value

    def start_experiment(self, job_type: Optional[str] = None, group: Optional[str] = None, *args, **kwargs):
        """Start experiment(logger post init or reinit next experiment with the same name).

        Parameters
        ----------
        job_type:
            Specify the type of run, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual runs into a larger experiment.
        """
        self.job_type = job_type
        self.group = group
        self.reinit_experiment()

    def reinit_experiment(self):
        """Reinit experiment."""
        self._experiment = wandb.init(
            name=self.name,
            project=self.project,
            entity=self.entity,
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
        from pytorch_lightning.loggers import WandbLogger as PLWandbLogger

        self._pl_logger = PLWandbLogger(experiment=self.experiment, log_model=self.log_model)
        return self._pl_logger

    @property
    def experiment(self):
        """Init experiment."""
        if self._experiment is None:
            self.reinit_experiment()
        return self._experiment
