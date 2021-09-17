import warnings

import numpy as np
import pandas as pd

from etna.models.base import PerSegmentModel


class _SeasonalMovingAverageModel:
    """
    Seasonal moving average.

    Forecast for point y_t is calculated as mean of y_{t - s}, y_{t - 2 * s}, ...,
    y_{t - n * s} where s is seasonality, n is window size (how many history values are taken for forecast).
    """

    def __init__(self, window: int = 5, seasonality: int = 7):
        """
        Initialize seasonal moving average model.

        Length of remembered tail of series is window * seasonality.

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

    def fit(self, df: pd.DataFrame) -> "_SeasonalMovingAverageModel":
        """
        Fitting simple model on given series.

        Parameters
        ----------
        df: pd.DataFrame
            Ignored. Needed for compatibility with AutoRegressorForecaster.

        Returns
        -------
        self: SeasonalMovingAverageModel
            fitted model
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

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate forecast.

        Parameters
        ----------
        df: pd.DataFrame
            Used only for getting the horizon of forecast. Needed for compatibility with AutoRegressorForecaster.
            len(features) = horizon.

        Returns
        -------
        pd.Series with forecast.
        """
        horizon = len(df)
        res = np.append(self.series, np.zeros(horizon))
        for i in range(self.shift, len(res)):
            res[i] = res[i - self.shift : i : self.seasonality].mean()
        return pd.Series(data=res[-horizon:], name=self.name)


class SeasonalMovingAverageModel(PerSegmentModel):
    """
    Seasonal moving average.

    Forecast for point y_t is calculated as mean of y_{t - s}, y_{t - 2 * s}, ...,
    y_{t - n * s} where s is seasonality, n is window size (how many history values are taken for forecast).
    """

    def __init__(self, window: int = 5, seasonality: int = 7):
        """
        Initialize seasonal moving average model.

        Length of remembered tail of series is window * seasonality.

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


__all__ = ["SeasonalMovingAverageModel"]
