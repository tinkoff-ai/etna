import warnings
from enum import Enum
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.mixins import NonPredictionIntervalContextRequiredModelMixin
from etna.models.mixins import PerSegmentModelMixin


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

    @staticmethod
    def _get_context_beginning(
        df: pd.DataFrame, prediction_size: int, seasonality: SeasonalityMode, window: int
    ) -> pd.Timestamp:
        """
        Get timestamp where context begins.

        Parameters
        ----------
        df:
            Time series in a long format.
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context for models that require it.
        seasonality:
            Seasonality.
        window:
            Number of values taken for forecast of each point.

        Returns
        -------
        :
            Timestamp with beginning of the context.

        Raises
        ------
        ValueError:
            if context isn't big enough
        """
        df_history = df.iloc[:-prediction_size]
        history_timestamps = df_history["timestamp"]
        future_timestamps = df["timestamp"].iloc[-prediction_size:]

        # if we have len(history_timestamps) == 0, then len(df) <= prediction_size
        if len(history_timestamps) == 0:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size of increase length of given dataframe!"
            )

        if seasonality is SeasonalityMode.month:
            first_index = future_timestamps.iloc[0] - pd.DateOffset(months=window)

        elif seasonality is SeasonalityMode.year:
            first_index = future_timestamps.iloc[0] - pd.DateOffset(years=window)

        if first_index < history_timestamps.iloc[0]:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size of increase length of given dataframe!"
            )

        return first_index

    def _make_predictions(self, result_template: pd.Series, context: pd.Series, prediction_size: int) -> np.ndarray:
        """Make predictions using ``result_template`` as a base and ``context`` as a context."""
        index = result_template.index
        start_idx = len(result_template) - prediction_size
        end_idx = len(result_template)
        for i in range(start_idx, end_idx):
            for w in range(1, self.window + 1):
                if self.seasonality == SeasonalityMode.month:
                    prev_date = result_template.index[i] - pd.DateOffset(months=w)
                elif self.seasonality == SeasonalityMode.year:
                    prev_date = result_template.index[i] - pd.DateOffset(years=w)

                result_template.loc[index[i]] += context.loc[prev_date]

            result_template.loc[index[i]] = result_template.loc[index[i]] / self.window

        result_values = result_template.values[-prediction_size:]
        return result_values

    def forecast(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """Compute autoregressive forecasts.

        Parameters
        ----------
        df:
            Features dataframe.
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
            if context isn't big enough
        ValueError:
            if forecast context contains NaNs
        """
        context_beginning = self._get_context_beginning(
            df=df, prediction_size=prediction_size, seasonality=self.seasonality, window=self.window
        )

        df = df.set_index("timestamp")
        df_history = df.iloc[:-prediction_size]
        history = df_history["target"]
        history = history[history.index >= context_beginning]
        if np.any(history.isnull()):
            raise ValueError("There are NaNs in a forecast context, forecast method required context to filled!")

        index = pd.date_range(start=context_beginning, end=df.index[-1], freq=self._freq)
        result_template = np.append(history.values, np.zeros(prediction_size))
        result_template = pd.Series(result_template, index=index)
        result_values = self._make_predictions(
            result_template=result_template, context=result_template, prediction_size=prediction_size
        )
        return result_values

    def predict(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """Compute predictions using true target data as context.

        Parameters
        ----------
        df:
            Features dataframe.
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
            if context isn't big enough
        ValueError:
            if there are NaNs in a target column on timestamps that are required to make predictions
        """
        context_beginning = self._get_context_beginning(
            df=df, prediction_size=prediction_size, seasonality=self.seasonality, window=self.window
        )

        df = df.set_index("timestamp")
        context = df["target"]
        context = context[context.index >= context_beginning]
        if np.any(np.isnan(context)):
            raise ValueError("There are NaNs in a target column, predict method requires target to be filled!")

        index = pd.date_range(start=df.index[-prediction_size], end=df.index[-1], freq=self._freq)
        result_template = pd.Series(np.zeros(prediction_size), index=index)
        result_values = self._make_predictions(
            result_template=result_template, context=context, prediction_size=prediction_size
        )
        return result_values

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
