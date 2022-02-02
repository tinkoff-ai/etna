from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd

from etna.core.mixins import BaseMixin

if TYPE_CHECKING:
    from etna.datasets import TSDataset


class BaseLogger(ABC, BaseMixin):
    """Abstract class for implementing loggers."""

    def __init__(self):
        """Create logger instance."""
        pass

    @abstractmethod
    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """
        Log any event.

        e.g. "Fitted segment segment_name"

        Parameters
        ----------
        msg:
            Message or dict to log
        kwargs:
            Parameters for changing additional info in log message
        """
        pass

    @abstractmethod
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
            Dataframe produced with TimeSeriesCrossValidation.get_metrics(aggregate_metrics=False)
        forecast_df
            Forecast from backtest
        fold_info_df:
            Fold information from backtest
        """
        pass

    def start_experiment(self, *args, **kwargs):
        """Start experiment(logger post init or reinit next experiment with the same name)."""
        pass

    def finish_experiment(self, *args, **kwargs):
        """Finish experiment."""
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
        pass


class _Logger(BaseLogger):
    """Composite for loggers."""

    def __init__(self):
        """Create instance for composite of loggers."""
        super().__init__()
        self.loggers = []

    def add(self, logger: BaseLogger) -> int:
        """
        Add new logger.

        Parameters
        ----------
        logger:
            logger to be added

        Returns
        -------
        result: int
            identifier of added logger
        """
        self.loggers.append(logger)
        return len(self.loggers) - 1

    def remove(self, idx: int):
        """
        Remove logger by identifier.

        Parameters
        ----------
        idx:
            identifier of added logger
        """
        self.loggers.pop(idx)

    def log(self, msg: Union[str, Dict[str, Any]], **kwargs):
        """Log any event."""
        for logger in self.loggers:
            logger.log(msg, **kwargs)

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
        for logger in self.loggers:
            logger.log_backtest_metrics(ts, metrics_df, forecast_df, fold_info_df)

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
        for logger in self.loggers:
            logger.log_backtest_run(metrics, forecast, test)

    def start_experiment(self, *args, **kwargs):
        """Start experiment(logger post init or reinit next experiment with the same name)."""
        for logger in self.loggers:
            logger.start_experiment(*args, **kwargs)

    def finish_experiment(self):
        """Finish experiment."""
        for logger in self.loggers:
            logger.finish_experiment()

    @property
    def pl_loggers(self):
        """Pytorch lightning loggers."""
        return [logger.pl_logger for logger in self.loggers if "_pl_logger" in vars(logger)]

    @contextmanager
    def disable(self):
        """Context manager for local logging disabling."""
        temp_loggers = self.loggers
        self.loggers = []
        yield
        self.loggers = temp_loggers


def percentile(n: int):
    """Percentile for pandas agg."""

    def percentile_(x):
        return np.nanpercentile(a=x.values, q=n)

    percentile_.__name__ = "percentile_%s" % n
    return percentile_


def aggregate_metrics_df(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate metrics in `log_backtest_metrics` method.

    Parameters
    ----------
    metrics_df:
        Dataframe produced with Pipeline._get_backtest_metrics()
    """
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

    return metrics_dict_wide
