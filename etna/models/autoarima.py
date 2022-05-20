import warnings
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
        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        try:
            df.loc[:, categorical_cols] = df[categorical_cols].astype(int)
        except ValueError:
            raise ValueError(
                f"Categorical columns {categorical_cols} can not been converted to int.\n "
                "Try to encode this columns manually."
            )

        self._check_df(df)

        targets = df["target"]
        targets.index = df["timestamp"]

        exog_train = self._select_regressors(df)

        self._model = pm.auto_arima(df["target"], X=exog_train, **self.kwargs)
        return self

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
        if self._model is None:
            raise ValueError("AutoARIMA model is not fitted! Fit the model before calling predict method!")
        horizon = len(df)
        self._check_df(df, horizon)

        categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
        try:
            df.loc[:, categorical_cols] = df[categorical_cols].astype(int)
        except ValueError:
            raise ValueError(
                f"Categorical columns {categorical_cols} can not been converted to int.\n "
                "Try to encode this columns manually."
            )

        exog_future = self._select_regressors(df)
        if prediction_interval:
            confints = np.unique([2 * i if i < 0.5 else 2 * (1 - i) for i in quantiles])

            y_pred = pd.DataFrame({"target": self._model.predict(len(df), X=exog_future), "timestamp": df["timestamp"]})

            for confint in confints:
                forecast = self._model.predict(len(df), X=exog_future, return_conf_int=True, alpha=confint)
                if confint / 2 in quantiles:
                    y_pred[f"target_{confint/2:.4g}"] = forecast[1][:, :1]
                if 1 - confint / 2 in quantiles:
                    y_pred[f"target_{1 - confint/2:.4g}"] = forecast[1][:, 1:]
        else:
            y_pred = pd.DataFrame({"target": self._model.predict(len(df), X=exog_future), "timestamp": df["timestamp"]})
        y_pred = y_pred.reset_index(drop=True, inplace=False)
        return y_pred

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
