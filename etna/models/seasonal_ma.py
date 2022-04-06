import warnings
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.models.base import PerSegmentModel


class _SeasonalMovingAverageModel:
    """
    Seasonal moving average.

    .. math::
        y_{t} = \\frac{\\sum_{i=1}^{n} y_{t-is} }{n},

    where :math:`s` is seasonality, :math:`n` is window size (how many history values are taken for forecast).
    """

    def __init__(self, window: int = 5, seasonality: int = 7):
        """
        Initialize seasonal moving average model.

        Length of remembered tail of series is ``window * seasonality``.

        Parameters
        ----------
        window: int
            Number of values taken for forecast for each point.
        seasonality: int
            Lag between values taken for forecast.
        """
        self.series = None
        self.name = "target"
        self.window = window
        self.seasonality = seasonality
        self.shift = self.window * self.seasonality

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_SeasonalMovingAverageModel":
        """
        Fit SeasonalMovingAverage model.

        Parameters
        ----------
        df: pd.DataFrame
            Data to fit on
        regressors:
            List of the columns with regressors(ignored in this model)

        Returns
        -------
        :
            Fitted model
        """
        if set(df.columns) != {"timestamp", "target"}:
            warnings.warn(
                message=f"{type(self).__name__} does not work with any exogenous series or features. "
                f"It uses only target series for predict/\n "
            )
        targets = df["target"]
        if len(targets) < self.shift:
            raise ValueError(
                "Given series is too short for chosen shift value. Try lower shift value, or give" "longer series."
            )
        self.series = targets[-self.shift :].values

        # ???
        if targets.name is not None:
            self.name = targets.name
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predictions from a SeasonalMovingAverage model.

        Parameters
        ----------
        df: pd.DataFrame
            Used only for getting the horizon of forecast

        Returns
        -------
        :
            Array with predictions.
        """
        horizon = len(df)
        res = np.append(self.series, np.zeros(horizon))
        for i in range(self.shift, len(res)):
            res[i] = res[i - self.shift : i : self.seasonality].mean()
        y_pred = res[-horizon:]
        return y_pred


class SeasonalMovingAverageModel(PerSegmentModel):
    """
    Seasonal moving average.

    .. math::
        y_{t} = \\frac{\\sum_{i=1}^{n} y_{t-is} }{n},

    where :math:`s` is seasonality, :math:`n` is window size (how many history values are taken for forecast).
    """

    def __init__(self, window: int = 5, seasonality: int = 7):
        """
        Initialize seasonal moving average model.

        Length of remembered tail of series is ``window * seasonality``.

        Parameters
        ----------
        window: int
            Number of values taken for forecast for each point.
        seasonality: int
            Lag between values taken for forecast.
        """
        self.window = window
        self.seasonality = seasonality
        super(SeasonalMovingAverageModel, self).__init__(
            base_model=_SeasonalMovingAverageModel(window=window, seasonality=seasonality)
        )

    def get_model(self) -> Dict[str, "SeasonalMovingAverageModel"]:
        """Get internal model.

        Returns
        -------
        :
           Internal model
        """
        return self._get_model()


__all__ = ["SeasonalMovingAverageModel"]
