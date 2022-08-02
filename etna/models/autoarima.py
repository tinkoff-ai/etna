import warnings
from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning

from etna.models.base import BaseAdapter
from etna.models.base import PerSegmentPredictionIntervalModel
from etna.models.utils import determine_num_steps

warnings.filterwarnings(
    message="No frequency information was provided, so inferred frequency .* will be used",
    action="ignore",
    category=ValueWarning,
    module="statsmodels.tsa.base.tsa_model",
)


class _AutoARIMAAdapter(BaseAdapter):
    """
    Class for holding auto arima model.

    Notes
    -----
    We use auto ARIMA [1] model from pmdarima package.

    .. `auto ARIMA: <https://alkaline-ml.com/pmdarima/>_`

    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Init auto ARIMA model with given params.

        Parameters
        ----------
        **kwargs:
            Training parameters for auto_arima from pmdarima package.
        """
        self.kwargs = kwargs
        self._model: Optional[ARIMA] = None
        self._first_train_timestamp = None
        self._last_train_timestamp = None
        self._freq = None
        self.regressor_columns: List[str] = []

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_AutoARIMAAdapter":
        """
        Fits auto ARIMA model.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors

        Returns
        -------
        :
            Fitted model
        """
        self.regressor_columns = regressors

        self._encode_categoricals(df)
        self._check_df(df)

        freq = pd.infer_freq(df["timestamp"], warn=False)
        if freq is None:
            raise ValueError("Can't determine frequency of a given dataframe")

        targets = df["target"]
        targets.index = df["timestamp"]

        exog_train = self._select_regressors(df)

        self._model = pm.auto_arima(df["target"], X=exog_train, **self.kwargs)
        self._first_train_timestamp = df["timestamp"].min()
        self._last_train_timestamp = df["timestamp"].max()
        self._freq = freq

        return self

    def _predict_by_func(
        self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float], predict_func: Callable
    ):
        exog_future = self._select_regressors(df)
        if prediction_interval:
            confints = np.unique([2 * i if i < 0.5 else 2 * (1 - i) for i in quantiles])

            y_pred = pd.DataFrame({"target": predict_func(X=exog_future)})

            for confint in confints:
                forecast = predict_func(X=exog_future, return_conf_int=True, alpha=confint)
                if confint / 2 in quantiles:
                    y_pred[f"target_{confint/2:.4g}"] = forecast[1][:, :1]
                if 1 - confint / 2 in quantiles:
                    y_pred[f"target_{1 - confint/2:.4g}"] = forecast[1][:, 1:]
        else:
            y_pred = pd.DataFrame({"target": predict_func(X=exog_future)})

        return y_pred

    def _predict_in_sample(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]):
        first_timestamp = df["timestamp"].min()
        last_timestamp = df["timestamp"].max()

        if first_timestamp < self._first_train_timestamp:
            raise ValueError(
                f"It isn't possible to make in-sample prediction before training data! First training timestamp: {self._first_train_timestamp}, first timestamp to predict: {first_timestamp}"
            )

        start_idx = determine_num_steps(
            start_timestamp=self._first_train_timestamp, end_timestamp=first_timestamp, freq=self._freq
        )
        end_idx = determine_num_steps(
            start_timestamp=self._first_train_timestamp, end_timestamp=last_timestamp, freq=self._freq
        )

        predict_func = partial(self._model.predict_in_sample, start=start_idx, end=end_idx)  # type: ignore
        y_pred = self._predict_by_func(
            df=df, prediction_interval=prediction_interval, quantiles=quantiles, predict_func=predict_func
        )
        return y_pred

    def _predict_out_sample(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]):
        last_timestamp = df["timestamp"].max()
        steps_to_forecast = determine_num_steps(
            start_timestamp=self._last_train_timestamp, end_timestamp=last_timestamp, freq=self._freq
        )
        steps_to_skip = steps_to_forecast - df.shape[0]

        predict_func = partial(self._model.predict, n_periods=steps_to_forecast)  # type: ignore
        y_pred = self._predict_by_func(
            df=df, prediction_interval=prediction_interval, quantiles=quantiles, predict_func=predict_func
        )
        y_pred = y_pred.iloc[steps_to_skip:]
        return y_pred

    def predict(self, df: pd.DataFrame, prediction_interval: bool, quantiles: Sequence[float]) -> pd.DataFrame:
        """
        Compute predictions from auto ARIMA model.

        Parameters
        ----------
        df:
            Features dataframe
        prediction_interval:
             If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution

        Returns
        -------
        :
            DataFrame with predictions
        """
        if self._model is None or self._freq is None:
            raise ValueError("AutoARIMA model is not fitted! Fit the model before calling predict method!")

        self._encode_categoricals(df)

        # predict in-sample
        df_in_sample = df[df["timestamp"] <= self._last_train_timestamp]
        if not df_in_sample.empty:
            y_pred_in_sample = self._predict_in_sample(
                df=df_in_sample, prediction_interval=prediction_interval, quantiles=quantiles
            )
        else:
            y_pred_in_sample = None

        # predict out-sample
        df_out_sample = df[df["timestamp"] > self._last_train_timestamp]
        if not df_out_sample.empty:
            y_pred_out_sample = self._predict_out_sample(
                df=df_out_sample, prediction_interval=prediction_interval, quantiles=quantiles
            )
        else:
            y_pred_out_sample = None

        # assemble results
        if y_pred_in_sample is None:
            y_pred = y_pred_out_sample
        elif y_pred_out_sample is None:
            y_pred = y_pred_in_sample
        else:
            y_pred = pd.concat([y_pred_in_sample, y_pred_out_sample])
        y_pred = y_pred.reset_index(drop=True)
        return y_pred

    def _encode_categoricals(self, df: pd.DataFrame) -> None:
        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        try:
            df.loc[:, categorical_cols] = df[categorical_cols].astype(int)
        except ValueError:
            raise ValueError(
                f"Categorical columns {categorical_cols} can not been converted to int.\n "
                "Try to encode this columns manually."
            )

    def _check_df(self, df: pd.DataFrame, horizon: Optional[int] = None):
        column_to_drop = [col for col in df.columns if col not in ["target", "timestamp"] + self.regressor_columns]
        if column_to_drop:
            warnings.warn(
                message=f"AutoARIMA model does not work with exogenous features (features unknown in future).\n "
                f"{column_to_drop} will be dropped"
            )
        if horizon:
            short_regressors = [regressor for regressor in self.regressor_columns if df[regressor].count() < horizon]
            if short_regressors:
                raise ValueError(
                    f"Regressors {short_regressors} are too short for chosen horizon value.\n "
                    "Try lower horizon value, or drop this regressors."
                )

    def _select_regressors(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.regressor_columns:
            exog_future = df[self.regressor_columns]
            exog_future.index = df["timestamp"]
        else:
            exog_future = None
        return exog_future

    def get_model(self) -> ARIMA:
        """Get internal pmdarima.arima.arima.ARIMA model that is used inside etna class.

        Returns
        -------
        :
           Internal model
        """
        return self._model


class AutoARIMAModel(PerSegmentPredictionIntervalModel):
    """
    Class for holding auto arima model.

    Notes
    -----
    We use :py:class:`pmdarima.arima.arima.ARIMA`.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Init auto ARIMA model with given params.

        Parameters
        ----------
        **kwargs:
            Training parameters for auto_arima from pmdarima package.
        """
        self.kwargs = kwargs
        super(AutoARIMAModel, self).__init__(
            base_model=_AutoARIMAAdapter(
                **self.kwargs,
            )
        )
