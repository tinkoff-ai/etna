from typing import Iterable
from typing import Union

from etna.loggers.base import Logger
from etna.loggers.base import LoggerComposite
from etna.models.seasonal_ma import SeasonalMovingAverageModel


class NaiveModel(SeasonalMovingAverageModel):
    """Naive model predicts t-th value of series with its (t - lag) value."""

    def __init__(self, lag: int = 1, logger: Union[Logger, Iterable[Logger]] = LoggerComposite()):
        """
        Init NaiveModel.

        Parameters
        ----------
        lag: int
            lag for new value prediction
        """
        self.lag = lag
        super().__init__(window=1, seasonality=lag, logger=logger)


__all__ = ["NaiveModel"]
