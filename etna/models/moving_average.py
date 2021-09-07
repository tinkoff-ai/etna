from etna.models.seasonal_ma import SeasonalMovingAverageModel


class MovingAverageModel(SeasonalMovingAverageModel):
    """MovingAverageModel averages previous series values to forecast future one."""

    def __init__(self, window: int = 5):
        """
        Init MovingAverageModel.

        Parameters
        ----------
        window: int
            number of history points to average
        """
        self.window = window
        super().__init__(window=window, seasonality=1)


__all__ = ["MovingAverageModel"]
