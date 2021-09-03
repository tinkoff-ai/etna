from etna.models.seasonal_ma import SeasonalMovingAverageModel


class NaiveModel(SeasonalMovingAverageModel):
    """Naive model predicts t-th value of series with its (t - lag) value."""

    def __init__(self, lag: int = 1):
        """
        Init NaiveModel.

        Parameters
        ----------
        lag: int
            lag for new value prediction
        """
        self.lag = lag
        super().__init__(window=1, seasonality=lag)


__all__ = ["NaiveModel"]
