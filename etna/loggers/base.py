from abc import ABC
from abc import abstractmethod
from typing import Iterable

import pandas as pd

from etna.core.mixins import BaseMixin


class Logger(ABC, BaseMixin):
    """Abstract class for implementing loggers."""

    def __init__(self):
        """Create logger instance."""
        pass

    @abstractmethod
    def log(self, msg: str):
        """
        Log any event.

        e.g. "Fitted segment segment_name"

        Parameters
        ----------
        msg: str
            Message to log
        """
        pass

    @abstractmethod
    def log_backtest_metrics(
        self, df: pd.DataFrame, metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

        Parameters
        ----------
        df:
            Dataframe to train
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

    def set_config(self, forecaster):
        """Pass forecaster config to loggers."""
        pass


class LoggerComposite(Logger):
    """Composite for loggers."""

    def __init__(self, *args):
        """Create instance for composite of loggers."""
        super().__init__()
        if args == (None,):
            self.loggers = []
        elif len(args) == 1 and isinstance(args[0], LoggerComposite):
            self.loggers = args[0].loggers
        elif len(args) == 1 and isinstance(args[0], Logger):
            self.loggers = [args[0]]
        elif isinstance(args, Iterable):
            self.loggers = flatten(args)
        else:
            self.loggers = []

    def add(self, logger: Logger) -> int:
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

    def log(self, msg: str):
        """Log any event."""
        for logger in self.loggers:
            logger.log(msg)

    def log_backtest_metrics(
        self, df: pd.DataFrame, metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame
    ):
        """
        Write metrics to logger.

        Parameters
        ----------
        df:
            Dataframe to train
        metrics_df:
            Dataframe produced with TimeSeriesCrossValidation.get_metrics(aggregate_metrics=False)
        forecast_df:
            Forecast from backtest
        fold_info_df:
            Fold information from backtest
        """
        for logger in self.loggers:
            logger.log_backtest_metrics(df, metrics_df, forecast_df, fold_info_df)

    def log_backtest_run(self, metrics: pd.DataFrame, forecast: pd.DataFrame, test: pd.DataFrame):
        """
        Backtest metrics to logger.

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


def flatten(*args):
    """Flatten nested args."""
    output = []
    for arg in args:
        if hasattr(arg, "__iter__"):
            output.extend(flatten(*arg))
        else:
            output.append(arg)
    return output
