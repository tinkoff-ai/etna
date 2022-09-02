import warnings
from enum import Enum
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredModelMixin
from etna.models.base import PerSegmentModelMixin


class SeasonalityMode(Enum):
    """Enum for seasonality mode for DeadlineMovingAverageModel."""

    month = "month"
    year = "year"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} seasonality allowed"
        )


class _DeadlineMovingAverageModel:
    """Moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
        """
        Initialize deadline moving average model.

        Length of the context is equal to the number of ``window`` months or years, depending on the ``seasonality``.

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
        self._freq = None

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
        ValueError
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

        self._freq = freq

        return self

    def _get_context_beginning(self, df: pd.DataFrame, prediction_size: int):
        df_history = df.iloc[:-prediction_size]
        history_timestamps = df_history["timestamp"]
        future_timestamps = df["timestamp"].iloc[-prediction_size:]

        if self.seasonality == SeasonalityMode.month:
            first_index = future_timestamps.iloc[0] - pd.DateOffset(months=self.window)

        elif self.seasonality == SeasonalityMode.year:
            first_index = future_timestamps.iloc[0] - pd.DateOffset(years=self.window)

        if first_index < history_timestamps.iloc[0]:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size of increase length of given dataframe!"
            )

        return first_index

    def predict(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """
        Compute predictions from a DeadlineMovingAverageModel.

        Parameters
        ----------
        df: pd.DataFrame
            Used only for getting the horizon of forecast and timestamps.
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
        context_beginning = self._get_context_beginning(df=df, prediction_size=prediction_size)

        df_history = df.iloc[:-prediction_size]
        history_targets = df_history["target"]
        history_timestamps = df_history["timestamp"]
        future_timestamps = df["timestamp"].iloc[-prediction_size:]
        history_targets = history_targets.loc[history_timestamps >= context_beginning]
        history_timestamps = history_timestamps.loc[history_timestamps >= context_beginning]

        index = pd.date_range(start=context_beginning, end=future_timestamps.iloc[-1])
        res = np.append(history_targets.values, np.zeros(prediction_size))
        res = pd.DataFrame(res)
        res.index = index
        for i in range(len(history_targets), len(res)):
            for w in range(1, self.window + 1):
                if self.seasonality == SeasonalityMode.month:
                    prev_date = res.index[i] - pd.DateOffset(months=w)
                elif self.seasonality == SeasonalityMode.year:
                    prev_date = res.index[i] - pd.DateOffset(years=w)

                if prev_date <= history_timestamps.iloc[-1]:
                    res.loc[index[i]] += history_targets.loc[history_timestamps == prev_date].values
                else:
                    res.loc[index[i]] += res.loc[prev_date].values

            res.loc[index[i]] = res.loc[index[i]] / self.window

        res = res.values.ravel()[-prediction_size:]
        return res

    @property
    def context_size(self) -> int:
        """Upper bound to context size of the model."""
        cur_value = None
        if self.seasonality is SeasonalityMode.year:
            cur_value = 366
        elif self.seasonality is SeasonalityMode.month:
            cur_value = 31

        if self._freq is None:
            raise ValueError("Model is not fitted! Fit the model before trying the find out context size!")
        if self._freq == "H":
            cur_value *= 24

        cur_value *= self.window

        return cur_value


class DeadlineMovingAverageModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextRequiredModelMixin,
    NonPredictionIntervalContextRequiredAbstractModel,
):
    """Moving average model that uses exact previous dates to predict."""

    def __init__(self, window: int = 3, seasonality: str = "month"):
        """
        Initialize deadline moving average model.

        Length of the context is equal to the number of ``window`` months or years, depending on the ``seasonality``.

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

    @property
    def context_size(self) -> int:
        """Upper bound to context size of the model."""
        models = self.get_model()
        model = next(iter(models.values()))
        return model.context_size

    def get_model(self) -> Dict[str, "DeadlineMovingAverageModel"]:
        """Get internal model.

        Returns
        -------
        :
           Internal model
        """
        return self._get_model()


__all__ = ["DeadlineMovingAverageModel"]
