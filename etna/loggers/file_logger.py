import datetime
import json
import os
import pathlib
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd

from etna.loggers.base import BaseLogger
from etna.loggers.base import percentile

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

    @abstractmethod
    def start_experiment(self, job_type: Optional[str] = None, group: Optional[str] = None, *args, **kwargs):
        """Start experiment within current experiment, it is used for separate different folds during backtest.

        Parameters
        ----------
        job_type:
            Specify the type of run, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual runs into a larger experiment.
        """
        pass

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

        self._save_table(metrics, "metrics")
        self._save_table(TSDataset.to_flatten(forecast), "forecast")
        self._save_table(TSDataset.to_flatten(test), "test")

        metrics_dict = (
            metrics.drop(["segment"], axis=1)
            .apply(["median", "mean", "std", percentile(5), percentile(25), percentile(75), percentile(95)])
            .to_dict()
        )

        metrics_dict_wide = {
            f"{metrics_key}_{statistics_key}": value
            for metrics_key, values in metrics_dict.items()
            for statistics_key, value in values.items()
        }

        self._save_dict(metrics_dict_wide, "metrics_summary")

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

        self._save_table(metrics_df, "metrics")
        self._save_table(TSDataset.to_flatten(forecast_df), "forecast")
        self._save_table(fold_info_df, "fold_info")

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
            for statistics_key, value in values.items()
        }

        self._save_dict(metrics_dict_wide, "metrics_summary")


class LocalFileLogger(BaseFileLogger):
    """Logger for logging files into local folder."""

    def __init__(self, experiments_folder: str):
        """
        Create instance of LocalFileLogger.

        Parameters
        ----------
        experiments_folder:
            path to folder to create experiment in

        Raises
        ------
        ValueError:
            if wrong path is given
        """
        super().__init__()
        if not os.path.isdir(experiments_folder):
            raise ValueError(f"Folder {experiments_folder} doesn't exist")
        self.experiments_folder = experiments_folder
        cur_datetime = datetime.datetime.now()
        subfolder_name = cur_datetime.strftime("%Y-%m-%d %H-%M-%S")
        self.experiment_folder = pathlib.Path(self.experiments_folder).joinpath(subfolder_name)
        self.experiment_folder.mkdir()
        self._current_experiment_folder: Optional[pathlib.Path] = None

    def start_experiment(self, job_type: Optional[str] = None, group: Optional[str] = None, *args, **kwargs):
        """Start experiment within current experiment, it is used for separate different folds during backtest.

        Parameters
        ----------
        job_type:
            Specify the type of run, which is useful when you're grouping runs together
            into larger experiments using group.
        group:
            Specify a group to organize individual runs into a larger experiment.
        """
        self._current_experiment_folder = self.experiment_folder.joinpath(f"{job_type}_{group}")
        self._current_experiment_folder.mkdir()

    def _save_table(self, table: pd.DataFrame, name: str):
        """Save table with given name.

        Parameters
        ----------
        table:
            dataframe to save
        name:
            filename without extensions
        """
        if self._current_experiment_folder is None:
            raise ValueError("You should start experiment before using log_backtest_run or log_backtest_metrics")
        filename = f"{name}.csv"
        table.to_csv(self._current_experiment_folder.joinpath(filename), index=False)

    def _save_dict(self, dictionary: Dict[str, Any], name: str):
        """Save dictionary with given name.

        Parameters
        ----------
        dictionary:
            dict to save
        name:
            filename without extensions
        """
        if self._current_experiment_folder is None:
            raise ValueError("You should start experiment before using log_backtest_run or log_backtest_metrics")
        filename = f"{name}.json"
        with open(self._current_experiment_folder.joinpath(filename), "w") as ouf:
            json.dump(dictionary, ouf)


# TODO: make S3 logger
