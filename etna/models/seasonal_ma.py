import warnings

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel


class SeasonalMovingAverageModel(
    NonPredictionIntervalContextRequiredAbstractModel,
):
    """Seasonal moving average.

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
        window:
            Number of values taken for forecast for each point.
        seasonality:
            Lag between values taken for forecast.
        """
        self.window = window
        self.seasonality = seasonality

    @property
    def context_size(self) -> int:
        """Context size of the model."""
        return self.window * self.seasonality

    def get_model(self) -> "SeasonalMovingAverageModel":
        """Get internal model.

        Returns
        -------
        :
           Itself
        """
        return self

    def fit(self, ts: TSDataset) -> "SeasonalMovingAverageModel":
        """Fit model.

        For this model, fit does nothing.

        Parameters
        ----------
        ts:
            Dataset with features

        Returns
        -------
        :
            Model after fit
        """
        columns = set(ts.columns.get_level_values("feature"))
        if columns != {"target"}:
            warnings.warn(
                message=f"{type(self).__name__} does not work with any exogenous series or features. "
                f"It uses only target series for predict/\n "
            )
        return self

    def _validate_context(self, df: pd.DataFrame, prediction_size: int):
        """Validate that we have enough context to make prediction with given parameters."""
        expected_length = prediction_size + self.context_size

        if len(df) < expected_length:
            raise ValueError(
                "Given context isn't big enough, try to decrease context_size, prediction_size or increase length of given dataframe!"
            )

    def _predict_components(self, df: pd.DataFrame, prediction_size: int) -> pd.DataFrame:
        """Estimate forecast components."""
        from etna.transforms import LagTransform

        self._validate_context(df=df, prediction_size=prediction_size)

        lag_transform = LagTransform(
            in_column="target",
            lags=list(range(self.seasonality, self.context_size + 1, self.seasonality)),
            out_column="target_component_lag",
        )
        target_components_df = lag_transform._transform(df) / self.window
        target_components_df = target_components_df.iloc[-prediction_size:]
        target_components_df = target_components_df.drop(columns=["target"], level="feature")
        return target_components_df

    def _forecast(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """Make autoregressive forecasts on a wide dataframe."""
        self._validate_context(df=df, prediction_size=prediction_size)

        expected_length = prediction_size + self.context_size
        history = df.loc[:, pd.IndexSlice[:, "target"]].values
        history = history[-expected_length:-prediction_size]
        if np.any(np.isnan(history)):
            raise ValueError("There are NaNs in a forecast context, forecast method requires context to be filled!")

        num_segments = history.shape[1]
        res = np.append(history, np.zeros((prediction_size, num_segments)), axis=0)
        for i in range(self.context_size, len(res)):
            res[i] = res[i - self.context_size : i : self.seasonality].mean(axis=0)

        y_pred = res[-prediction_size:]
        return y_pred

    def forecast(self, ts: TSDataset, prediction_size: int, return_components: bool = False) -> TSDataset:
        """Make autoregressive forecasts.

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions

        Raises
        ------
        NotImplementedError:
            if return_components mode is used
        ValueError:
            if context isn't big enough
        ValueError:
            if forecast context contains NaNs
        """
        df = ts.to_pandas()
        y_pred = self._forecast(df=df, prediction_size=prediction_size)
        ts.df = ts.df.iloc[-prediction_size:]
        ts.df.loc[:, pd.IndexSlice[:, "target"]] = y_pred

        if return_components:
            df.loc[-prediction_size:, pd.IndexSlice[:, "target"]] = y_pred
            target_components_df = self._predict_components(df=df, prediction_size=prediction_size)
            ts.add_target_components(target_components_df=target_components_df)
        return ts

    def _predict(self, df: pd.DataFrame, prediction_size: int) -> np.ndarray:
        """Make predictions on a wide dataframe using true values as autoregression context."""
        self._validate_context(df=df, prediction_size=prediction_size)

        expected_length = prediction_size + self.context_size
        context = df.loc[:, pd.IndexSlice[:, "target"]].values
        context = context[-expected_length:]
        if np.any(np.isnan(context)):
            raise ValueError("There are NaNs in a target column, predict method requires target to be filled!")

        num_segments = context.shape[1]
        res = np.zeros((prediction_size, num_segments))
        for res_idx, context_idx in enumerate(range(self.context_size, len(context))):
            res[res_idx] = context[context_idx - self.context_size : context_idx : self.seasonality].mean(axis=0)

        y_pred = res[-prediction_size:]
        return y_pred

    def predict(self, ts: TSDataset, prediction_size: int, return_components: bool = False) -> TSDataset:
        """Make predictions using true values as autoregression context (teacher forcing).

        Parameters
        ----------
        ts:
            Dataset with features
        prediction_size:
            Number of last timestamps to leave after making prediction.
            Previous timestamps will be used as a context.
        return_components:
            If True additionally returns prediction components

        Returns
        -------
        :
            Dataset with predictions

        Raises
        ------
        NotImplementedError:
            if return_components mode is used
        ValueError:
            if context isn't big enough
        ValueError:
            if forecast context contains NaNs
        """
        df = ts.to_pandas()
        y_pred = self._predict(df=df, prediction_size=prediction_size)
        ts.df = ts.df.iloc[-prediction_size:]
        ts.df.loc[:, pd.IndexSlice[:, "target"]] = y_pred

        if return_components:
            target_components_df = self._predict_components(df=df, prediction_size=prediction_size)
            ts.add_target_components(target_components_df=target_components_df)
        return ts


__all__ = ["SeasonalMovingAverageModel"]
