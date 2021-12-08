from typing import Union, Dict, Any, TYPE_CHECKING
from abc import abstractmethod
import json
import os
import pathlib

import pandas as pd
import datetime

from etna.loggers.base import BaseLogger, percentile

if TYPE_CHECKING:
    from etna.datasets import TSDataset


class BaseFileLogger(BaseLogger):
    """Base logger for logging files."""

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """
        Log any event.

        This class does nothing with it, use other loggers to do it.

        Parameters
        ----------
        msg:
            Message or dict to log
        kwargs:
            Parameters for changing additional info in log message
        """
        pass

    @abstractmethod
    def _save_table(self, table: pd.DataFrame, name: str):
        """Save table with given name.

        Parameters
        ----------
        table:
            dataframe to save
        name:
            filename without extensions
        """
        pass

    @abstractmethod
    def _save_dict(self, dictionary: Dict[str, Any], name: str):
        """Save dictionary with given name.

        Parameters
        ----------
        dictionary:
            dict to save
        name:
            filename without extensions
        """
        pass

    def log_backtest_metrics(
        self, ts: "TSDataset", metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

        Parameters
        ----------
        ts:
            TSDataset to with backtest data
        metrics_df:
            Dataframe produced with Pipeline._get_backtest_metrics()
        forecast_df:
            Forecast from backtest
        fold_info_df:
            Fold information from backtest
        """
        from etna.datasets import TSDataset

        self.save_table(metrics_df, "metrics")
        self.save_table(TSDataset.to_flatten(forecast_df), "forecast")
        self.save_table(TSDataset.to_flatten(fold_info_df), "fold_info_df")

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

        metrics_dict_wide = {
            f"{metrics_key}_{statistics_key}": value
            for metrics_key, values in metrics_dict.items()
            for statistics_key, value in values.itmes()
        }

        self.save_dict(metrics_dict_wide, "metrics_summary")

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

        self.save_table(metrics, "metrics")
        self.save_table(TSDataset.to_flatten(forecast), "forecast")
        self.save_table(TSDataset.to_flatten(test), "test")

        metrics_dict = (
            metrics.drop(["segment"], axis=1)
            .apply(["median", "mean", "std", percentile(5), percentile(25), percentile(75), percentile(95)])
            .to_dict()
        )

        metrics_dict_wide = {
            f"{metrics_key}_{statistics_key}": value
            for metrics_key, values in metrics_dict.items()
            for statistics_key, value in values.itmes()
        }

        self.save_dict(metrics_dict_wide, "metrics_fold_summary")


class LocalFileLogger(BaseFileLogger):
    """Logger for logging files into local folder."""

    def __init__(self, experiment_folder: str):
        """
        Create instance of LocalFileLogger.

        Parameters
        ----------
        experiment_folder:
            path to folder to create experiment in

        Raises
        ------
        ValueError:
            if wrong path is given
        """
        super().__init__()
        if not os.path.isdir(experiment_folder):
            raise ValueError(f"Folder {experiment_folder} doesn't exist")
        self.experiment_folder = experiment_folder
        cur_datetime = datetime.datetime.now()
        subfolder_name = cur_datetime.strftime("%Y-%m-%d %H-%M-%S")
        self.experiment_subfolder = pathlib.Path(self.experiment_folder).joinpath(subfolder_name)
        self.experiment_subfolder.mkdir()

    def _save_table(self, table: pd.DataFrame, name: str):
        """Save table with given name.

        Parameters
        ----------
        table:
            dataframe to save
        name:
            filename without extensions
        """
        filename = f"{name}.csv"
        table.to_csv(filename)

    def _save_dict(self, dictionary: Dict[str, Any], name: str):
        """Save dictionary with given name.

        Parameters
        ----------
        dictionary:
            dict to save
        name:
            filename without extensions
        """
        filename = f"{name}.json"
        with open(filename, "w") as ouf:
            json.dump(dictionary, ouf)


# TODO: make S3 logger
