import warnings
from enum import Enum
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.models.base import PerSegmentModel


class SeasonalityMode(Enum):
    """Enum for seasonality mode for DeadlineMovingAverageModel."""

    month = "month"
    year = "year"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} seasonality allowed."
        )


class _DeadlineMovingAverageModel:
    """Moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
        """
        Initialize deadline moving average model.

        Length of remembered tail of series is ``window * seasonality``.

        Parameters
        ----------
        window: int
            Number of values taken for forecast for each point.
        seasonality: str
            Only allowed monthly or annual seasonality.
        """
        self.name = "target"
        self.window = window
        self.seasonality = SeasonalityMode(seasonality)
        self.freqs_available = {"H", "D"}

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_DeadlineMovingAverageModel":
        """
        Fit DeadlineMovingAverageModel model.

        Parameters
        ----------
        df: pd.DataFrame
            Data to fit on
        regressors:
            List of the columns with regressors(ignored in this model)

        Raises
        ------
        ValueError
            If freq of dataframe is not supported
            If series is too short for chosen shift value
        Returns
        -------
        :
            Fitted model
        """
        freq = pd.infer_freq(df["timestamp"])
        if freq not in self.freqs_available:
            raise ValueError(f"{freq} is not supported! Use daily or hourly frequency!")

        if set(df.columns) != {"timestamp", "target"}:
            warnings.warn(
                message=f"{type(self).__name__} does not work with any exogenous series or features. "
                f"It uses only target series for predict/\n "
            )
        targets = df["target"]
        self.timestamps = df["timestamp"]

        if self.seasonality == SeasonalityMode.month:
            first_index = self.timestamps.iloc[-1] - pd.DateOffset(months=self.window)

        elif self.seasonality == SeasonalityMode.year:
            first_index = self.timestamps.iloc[-1] - pd.DateOffset(years=self.window)

        if first_index < self.timestamps.iloc[0]:
            raise ValueError(
                "Given series is too short for chosen shift value. Try lower shift value, or give" "longer series."
            )

        self.series = targets.loc[self.timestamps >= first_index]
        self.shift = len(self.series)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
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
        timestamps = df["timestamp"]
        res = np.zeros((len(df), 1))
        for w in range(1, self.window + 1):
            if self.seasonality == SeasonalityMode.month:
                prev_dates = timestamps - pd.DateOffset(months=w)
            elif self.seasonality == SeasonalityMode.year:
                prev_dates = timestamps - pd.DateOffset(years=w)
            values = []
            for date in prev_dates:
                value = self.series.loc[self.timestamps == date].values
                values.append(value)

            res += np.array(values)

        res = res / self.window
        res = res.reshape(
            len(df),
        )
        return res


class DeadlineMovingAverageModel(PerSegmentModel):
    """Moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
        """
        Initialize deadline moving average model.

        Length of remembered tail of series is ``window * seasonality``.

        Parameters
        ----------
        window: int
            Number of values taken for forecast for each point.
        seasonality: str
            Only allowed monthly or annual seasonality.
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
