from etna.models.seasonal_ma import SeasonalMovingAverageModel


class MovingAverageModel(SeasonalMovingAverageModel):
    """MovingAverageModel averages previous series values to forecast future one.

    .. math::
        y_{t} = \\frac{\\sum_{i=1}^{n} y_{t-i} }{n},

    where :math:`n` is window size.

    Notes
    -----
    This model supports in-sample and out-of-sample prediction decomposition.
    Prediction components are corresponding target lags with weights of :math:`1/window`.
    """

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
