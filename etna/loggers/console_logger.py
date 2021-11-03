import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Union

import pandas as pd
from loguru import logger as _logger

from etna.loggers.base import BaseLogger

if TYPE_CHECKING:
    from etna.datasets import TSDataset


class ConsoleLogger(BaseLogger):
    """Log any events and metrics to stderr output. Uses loguru."""

    def __init__(self):
        """Create instance of ConsoleLogger."""
        super().__init__()
        if 0 in _logger._core.handlers:
            _logger.remove(0)
        _logger.add(sink=sys.stderr)
        self.logger = _logger.opt(depth=2, lazy=True, colors=True)

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
        self.logger.patch(lambda r: r.update(**kwargs)).info(msg)

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
            Dataframe produced with Pipeline._get_backtest_metrics() or TimeSeriesCrossValidation.get_metrics()
        forecast_df:
            Forecast from backtest
        fold_info_df:
            Fold information from backtest

        Notes
        -----
        The result of logging will be different for aggregate_metrics=True and aggregate_metrics=False
        """
        for _, row in metrics_df.iterrows():
            for metric in metrics_df.columns[1:-1]:
                # case for aggregate_metrics=False
                if "fold_number" in row:
                    msg = f'Fold {row["fold_number"]}:{row["segment"]}:{metric} = {row[metric]}'
                # case for aggregate_metrics=True
                else:
                    msg = f'Segment {row["segment"]}:{metric} = {row[metric]}'
                self.logger.info(msg)

    @property
    def pl_logger(self):
        """Pytorch lightning loggers."""
        return self._pl_logger
