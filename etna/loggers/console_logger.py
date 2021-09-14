import sys

import pandas as pd
from loguru import logger

from etna.loggers.base import Logger


class ConsoleLogger(Logger):
    """Log any events and metrics to stderr output. Uses loguru."""

    def __init__(self):
        """Create instance of ConsoleLogger."""
        super().__init__()
        if 0 in logger._core.handlers:
            logger.remove(0)
        logger.add(sink=sys.stderr)

    def log(self, msg: str):
        """
        Log any event.

        e.g. "Fitted segment segment_name" to stderr output.

        Parameters
        ----------
        msg: str
            Message to log
        """
        logger.opt(depth=1, lazy=True, colors=True).info(msg)

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
            Fold inforamtion from backtest
        """
        for _, row in metrics_df.iterrows():
            for metric in metrics_df.columns[1:-1]:
                msg = f'Fold {row["fold_number"]}:{row["segment"]}:{metric} = {row[metric]}'
                logger.opt(depth=1, lazy=True, colors=True).info(msg)

    @property
    def pl_logger(self):
        """Pytorch lightning loggers."""
        return self._pl_logger
