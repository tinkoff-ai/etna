from typing import Iterable
from typing import Union

from etna.loggers.base import Logger
from etna.loggers.base import LoggerComposite
from etna.models.seasonal_ma import SeasonalMovingAverageModel


class MovingAverageModel(SeasonalMovingAverageModel):
    """MovingAverageModel averages previous series values to forecast future one."""

    def __init__(self, window: int = 5, logger: Union[Logger, Iterable[Logger]] = LoggerComposite()):
        """
        Init MovingAverageModel.

        Parameters
        ----------
        window: int
            number of history points to average
        """
        self.window = window
        super().__init__(window=window, seasonality=1, logger=logger)


__all__ = ["MovingAverageModel"]
