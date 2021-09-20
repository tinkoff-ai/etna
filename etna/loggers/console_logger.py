import sys
from typing import Any
from typing import Dict
from typing import Union
from typing import Optional

import pandas as pd
from loguru import logger as _logger

from etna.loggers.base import BaseLogger


class ConsoleLogger(BaseLogger):
    """Log any events and metrics to stderr output. Uses loguru."""

    def __init__(self):
        """Create instance of ConsoleLogger."""
        super().__init__()
        if 0 in _logger._core.handlers:
            _logger.remove(0)
        _logger.add(sink=sys.stderr)
        self.logger = _logger.opt(depth=1, lazy=True, colors=True)

    def log(self, msg: Union[str, Dict[str, Any]], name: Optional[str] = None):
        """
        Log any event.

        e.g. "Fitted segment segment_name" to stderr output.

        Parameters
        ----------
        msg:
            Message or dict to log
        name:
            Name of function to show
        """
        if name is not None:
            self.logger.patch(lambda r: r.update(function=name)).info(msg)
        else:
            self.logger.info(msg)

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
                self.logger.info(msg)

    @property
    def pl_logger(self):
        """Pytorch lightning loggers."""
        return self._pl_logger
