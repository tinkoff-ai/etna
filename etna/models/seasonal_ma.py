import warnings
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredModelMixin
from etna.models.base import PerSegmentModelMixin


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

        Length of the context is ``window * seasonality``.

        Parameters
        ----------
        window: int
            Number of values taken for forecast for each point.
        seasonality: int
            Lag between values taken for forecast.
        """
        self.name = "target"
        self.window = window
        self.seasonality = seasonality
        self.shift = self.window * self.seasonality

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_SeasonalMovingAverageModel":
        """
        Fit SeasonalMovingAverage model.

        Parameters
        ----------
        df:
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

        return self

    def predict(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """
        Compute predictions from a SeasonalMovingAverage model.

        Parameters
        ----------
        df:
            Used only for getting the horizon of forecast
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.

        Returns
        -------
        :
            Array with predictions.

        Raises
        ------
        ValueError:
            if context isn't big enought
        """
        expected_length = prediction_size + self.shift
        if len(df) < expected_length:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size of increase length of given dataframe!"
            )

        history = df["target"][-expected_length:-prediction_size]
        res = np.append(history, np.zeros(prediction_size))
        for i in range(self.shift, len(res)):
            res[i] = res[i - self.shift : i : self.seasonality].mean()
        y_pred = res[-prediction_size:]
        return y_pred


class SeasonalMovingAverageModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextRequiredModelMixin,
    NonPredictionIntervalContextRequiredAbstractModel,
):
    """
    Seasonal moving average.

    .. math::
        y_{t} = \\frac{\\sum_{i=1}^{n} y_{t-is} }{n},

    where :math:`s` is seasonality, :math:`n` is window size (how many history values are taken for forecast).
    """

    def __init__(self, window: int = 5, seasonality: int = 7):
        """
        Initialize seasonal moving average model.

        Length of the context is ``window * seasonality``.

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

    @property
    def context_size(self) -> int:
        """Context size of the model."""
        return self.window * self.seasonality

    def get_model(self) -> Dict[str, "SeasonalMovingAverageModel"]:
        """Get internal model.

        Returns
        -------
        :
           Internal model
        """
        return self._get_model()


__all__ = ["SeasonalMovingAverageModel"]
