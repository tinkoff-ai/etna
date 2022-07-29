import warnings
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.models.base import PerSegmentModel


class _DeadlineMovingAverageModel:
    """Seasonal moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
        """
        Initialize deadline moving average model.

        Length of remembered tail of series is ``window * seasonality``.

        Parameters
        ----------
        window: int
            Number of values taken for forecast for each point.
        seasonality: str
            Monthly or annual seasonality.
        """
        self.name = "target"
        self.window = window
        if seasonality not in ["month", "year"]:
            raise ValueError("Incorrect type of seasonality")
        self.seasonality = seasonality

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_DeadlineMovingAverageModel":
        """
        Fit DeadlineMovingAverageModel model.

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
        freq = pd.infer_freq(df["timestamp"])
        if freq not in {"H", "D"}:
            warnings.warn(message=f"{type(freq)} is not supported! Use daily or hourly frequency!")
        if set(df.columns) != {"timestamp", "target"}:
            warnings.warn(
                message=f"{type(self).__name__} does not work with any exogenous series or features. "
                f"It uses only target series for predict/\n "
            )
        targets = df["target"]
        self.dates = df["timestamp"]
        if self.seasonality == "month":
            first_index = self.dates.iloc[-1] - pd.DateOffset(months=self.window)
            if first_index < self.dates.iloc[0]:
                raise ValueError(
                    "Given series is too short for chosen shift value. Try lower shift value, or give" "longer series."
                )
        else:
            first_index = self.dates.iloc[-1] - pd.DateOffset(years=self.window)
            if first_index < self.dates.iloc[0]:
                raise ValueError(
                    "Given series is too short for chosen shift value. Try lower shift value, or give" "longer series."
                )

        self.series = targets.loc[self.dates >= first_index]
        self.shift = len(self.series)

        # ???
        if targets.name is not None:
            self.name = targets.name
        return self

    def predict(self, df: pd.DataFrame):
        """
        Compute predictions from a DeadlineMovingAverageModel.

        Parameters
        ----------
        df: pd.DataFrame
            Used only for getting the horizon of forecast and timestamps

        Returns
        -------
        :
            Array with predictions.
        """
        timestamp = df["timestamp"]
        res = np.zeros((len(df), 1))
        for w in range(1, self.window + 1):
            prev_dates = []
            if self.seasonality == "month":
                prev_dates = timestamp - pd.DateOffset(months=w)
            else:
                prev_dates = timestamp - pd.DateOffset(years=w)
            values = []
            for date in prev_dates:
                value = self.series.loc[self.dates == date].values
                values.append(value)
            res += np.array(values)

        res = res / self.window
        res = res.reshape(
            len(df),
        )
        return res


class DeadlineMovingAverageModel(PerSegmentModel):
    """Seasonal moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
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
        super(DeadlineMovingAverageModel, self).__init__(
            base_model=_DeadlineMovingAverageModel(window=window, seasonality=seasonality)
        )

    def get_model(self) -> Dict[str, "DeadlineMovingAverageModel"]:
        """Get internal model.

        Returns
        -------
        :
           Internal model
        """
        return self._get_model()


__all__ = ["DeadlineMovingAverageModel"]
