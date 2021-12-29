import inspect
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel
from joblib import delayed
from scipy.stats import norm

from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode
from etna.models.base import Model
from etna.pipeline.base import PipelineBase, BacktestMixin, IntervalMixin
from etna.transforms.base import Transform


class CrossValidationMode(Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"


class Pipeline(BacktestMixin, IntervalMixin):
    """Pipeline of transforms with a final estimator."""

    support_prediction_interval = True

    def __init__(
        self,
        model: Model,
        transforms: Sequence[Transform] = (),
        horizon: int = 1,
        quantiles: Sequence[float] = (0.025, 0.975),
        n_folds: int = 3,
    ):
        """
        Create instance of Pipeline with given parameters.

        Parameters
        ----------
        model:
            Instance of the etna Model
        transforms:
            Sequence of the transforms
        horizon:
            Number of timestamps in the future for forecasting
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation

        Raises
        ------
        ValueError:
            If the horizon is less than 1, quantile is out of (0,1) or n_folds is less than 2.
        """
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.horizon = self._validate_horizon(horizon)
        self.n_folds = self._validate_cv(n_folds)
        self.quantiles = self._validate_quantiles(quantiles)
        self.ts: Optional[TSDataset] = None

    @staticmethod
    def _validate_cv(cv: int) -> int:
        """Check that given number of folds is grater than 1."""
        if cv > 1:
            return cv
        else:
            raise ValueError("At least two folds for backtest are expected.")

    def fit(self, ts: TSDataset) -> "Pipeline":
        """Fit the Pipeline.
        Fit and apply given transforms to the data, then fit the model on the transformed data.

        Parameters
        ----------
        ts:
            Dataset with timeseries data

        Returns
        -------
        Pipeline:
            Fitted Pipeline instance
        """
        self.ts = ts
        self.ts.fit_transform(self.transforms)
        self.model.fit(self.ts)
        self.ts.inverse_transform()
        return self

    def _forecast_prediction_interval(self, future: TSDataset) -> TSDataset:
        """Forecast prediction interval for the future."""
        if self.ts is None:
            raise ValueError("Pipeline is not fitted! Fit the Pipeline before calling forecast method.")
        _, forecasts, _ = self.backtest(self.ts, metrics=[MAE()], n_folds=self.n_folds)
        forecasts = TSDataset(df=forecasts, freq=self.ts.freq)
        residuals = (
            forecasts.loc[:, pd.IndexSlice[:, "target"]]
            - self.ts[forecasts.index.min() : forecasts.index.max(), :, "target"]
        )

        predictions = self.model.forecast(ts=future)
        se = scipy.stats.sem(residuals)
        borders = []
        for quantile in self.quantiles:
            z_q = norm.ppf(q=quantile)
            border = predictions[:, :, "target"] + se * z_q
            border.rename({"target": f"target_{quantile:.4g}"}, inplace=True, axis=1)
            borders.append(border)

        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

        return predictions

    def forecast(self, prediction_interval: bool = False) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast

        Returns
        -------
        TSDataset
            TSDataset with forecast
        """
        if self.ts is None:
            raise ValueError("Pipeline is not fitted! Fit the Pipeline before calling forecast method.")
        self.check_support_prediction_interval(prediction_interval)

        future = self.ts.make_future(self.horizon)
        if prediction_interval:
            if "prediction_interval" in inspect.signature(self.model.forecast).parameters:
                predictions = self.model.forecast(
                    ts=future, prediction_interval=prediction_interval, quantiles=self.quantiles
                )
            else:
                predictions = self._forecast_prediction_interval(future=future)
        else:
            predictions = self.model.forecast(ts=future)
        return predictions
