import warnings
from copy import deepcopy
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.pipeline import Pipeline


class StackingEnsemble(Pipeline):
    """StackingEnsemble is a pipeline that forecast future using the metamodel to combine the forecasts of the base models."""

    def __init__(
        self,
        pipelines: List[Pipeline],
        final_model: RegressorMixin = LinearRegression(),
        cv: Union[None, int] = None,
        features_to_use: Union[None, Literal[all], List[str]] = None,
        n_jobs: int = 1,
    ):
        """Init StackingEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble.
        final_model:
            Regression model with fit/predict interface which will be used to combine the base estimators.
        cv:
            Number of folds to use in the backtest. Backtest is not used for model evaluation but for prediction.
        features_to_use:
            Features except the forecasts of the base models to use in the `final_model`.
        n_jobs:
            Number of jobs to run in parallel.

        Raises
        ------
        ValueError:
            If the number of the pipelines is less than 2 or pipelines have different horizons.
        """
        self._validate_pipeline_number(pipelines=pipelines)
        self.pipelines = pipelines
        self.horizon = self._get_horizon(pipelines=pipelines)
        self.final_model = final_model
        self.cv = self._validate_cv(cv=cv)
        self._features_to_use = features_to_use
        self.features_to_use: Union[None, Set[str]] = None
        self.n_jobs = n_jobs

    @staticmethod
    def _validate_pipeline_number(pipelines: List[Pipeline]):
        """Check that given valid number of pipelines."""
        if len(pipelines) < 2:
            raise ValueError("At least two pipelines are expected.")

    @staticmethod
    def _get_horizon(pipelines: List[Pipeline]) -> int:
        """Get ensemble's horizon."""
        horizons = set([pipeline.horizon for pipeline in pipelines])
        if len(horizons) > 1:
            raise ValueError("All the pipelines should have the same horizon.")
        return horizons.pop()

    @staticmethod
    def _validate_cv(cv: Optional[int]) -> int:
        """Check that given number of folds is grater than 1."""
        if cv is None:
            return 3
        elif isinstance(cv, int):
            if cv < 2:
                raise ValueError("At least two folds for backtest are expected.")
            return cv
        else:
            raise ValueError("Invalid format for cv parameter. The cv could be None or Number.")

    def _get_features_to_use(self, forecasts: List[TSDataset]):
        """Return all the features from `_features_to_use` which can be obtained from base models' forecasts."""
        features_df = pd.concat([forecast.df for forecast in forecasts], axis=1)
        available_features = set(features_df.columns.get_level_values("feature")) - {"fold_number"}
        features_to_use = self._features_to_use
        if isinstance(features_to_use, list):
            features_to_use = set(features_to_use)
            if len(features_to_use) == 0:
                features_to_use = None
            elif features_to_use.issubset(available_features):
                pass
            else:
                unavailable_features = features_to_use - available_features
                warnings.warn(f"Features {unavailable_features} are not found and will be dropped!")
                features_to_use = features_to_use.intersection(available_features)
        elif features_to_use == "all":
            features_to_use = available_features - {"target"}
        elif features_to_use is None:
            pass
        else:
            warnings.warn(
                "Feature list is passed in the wrong format."
                "Only the base models' forecasts will be used for the final forecast."
            )
            features_to_use = None
        return features_to_use

    @staticmethod
    def _fit_pipeline(pipeline: Pipeline, ts: TSDataset) -> Pipeline:
        """Fit given pipeline with ts."""
        tslogger.log(msg=f"Start fitting {pipeline}.")
        pipeline.fit(ts=ts)
        tslogger.log(msg=f"Pipeline {pipeline} is fitted.")
        return pipeline

    def _backtest_pipeline(self, pipeline: Pipeline, ts: TSDataset) -> TSDataset:
        """Get forecasts from backtest for given pipeline."""
        _, forecasts, _ = pipeline.backtest(ts, metrics=[MAE()], n_folds=self.cv)
        forecasts = TSDataset(df=forecasts, freq=ts.freq)
        return forecasts

    def fit(self, ts: TSDataset) -> "StackingEnsemble":
        """Fit the ensemble.

        Parameters
        ----------
        ts:
            TSDataset to fit ensemble.

        Returns
        -------
        StackingEnsemble:
            Fitted ensemble.
        """
        self.ts = ts

        # Get forecasts from base models on backtest to fit the final model on
        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )

        # Fit the final model
        self.features_to_use = self._get_features_to_use(forecasts)
        x, y = self._make_features(forecasts=forecasts, train=True)
        self.final_model.fit(x, y)

        # Fit the base models
        self.pipelines = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._fit_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )
        return self

    def _make_features(
        self, forecasts: List[TSDataset], train: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.Series]:
        """Prepare features for the `final_model`."""
        # Stack targets from the forecasts
        targets = [
            forecast[:, :, "target"].rename({"target": f"regressor_target_{i}"}, axis=1)
            for i, forecast in enumerate(forecasts)
        ]
        targets = pd.concat(targets, axis=1)

        # Get features from features_to_use
        features = pd.DataFrame()
        if self.features_to_use is not None:
            features_left = self.features_to_use.copy()
            features_in_forecasts = [set(forecast.columns.get_level_values("feature")) for forecast in forecasts]
            unique_features_in_forecasts = []
            for features_in_forecast in features_in_forecasts:
                features_new = features_left.intersection(features_in_forecast)
                unique_features_in_forecasts.append(features_new)
                features_left -= features_new
            features = pd.concat(
                [forecast[:, :, unique_features_in_forecasts[i]] for i, forecast in enumerate(forecasts)], axis=1
            )

        features_df = pd.concat([features, targets], axis=1)

        # Flatten the features to fit the sklearn interface
        x = pd.concat([features_df.loc[:, segment] for segment in self.ts.segments], axis=0)
        if train:
            y = pd.concat(
                [
                    self.ts[forecasts[0].index.min() : forecasts[0].index.max(), segment, "target"]
                    for segment in self.ts.segments
                ],
                axis=0,
            )
            return x, y
        else:
            return x

    @staticmethod
    def _forecast_pipeline(pipeline: Pipeline) -> TSDataset:
        """Make forecast with given pipeline."""
        tslogger.log(msg=f"Start forecasting with {pipeline}.")
        forecast = pipeline.forecast()
        tslogger.log(msg=f"Forecast is done with {pipeline}.")
        return forecast

    def forecast(self) -> TSDataset:
        """Forecast with ensemble: compute the combination of pipelines' forecasts using `final_model`.

        Returns
        -------
        TSDataset:
            Dataset with forecasts.
        """
        # Get forecast
        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines
        )
        x = self._make_features(forecasts=forecasts, train=False)
        y = self.final_model.predict(x).reshape(self.horizon, -1)

        # Format the forecast into TSDataset
        for i, segment in enumerate(self.ts.segments):
            x.loc[i * self.horizon : (i + 1) * self.horizon, "segment"] = segment
        x.loc[:, "timestamp"] = x.index.values
        df_exog = TSDataset.to_dataset(x)

        df = forecasts[0][:, :, "target"].copy()
        df.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = np.NAN

        forecast = TSDataset(df=df, freq=self.ts.freq, df_exog=df_exog)
        forecast.loc[pd.IndexSlice[:], pd.IndexSlice[:, "target"]] = y
        return forecast
