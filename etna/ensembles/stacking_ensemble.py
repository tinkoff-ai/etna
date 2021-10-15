import warnings
from copy import deepcopy
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.models import LinearPerSegmentModel
from etna.models.base import Model
from etna.pipeline import Pipeline


class StackingEnsemble(Pipeline):
    """StackingEnsemble is a pipeline that forecast future using the metamodel to combine the forecasts of the base models."""

    def __init__(
        self,
        pipelines: List[Pipeline],
        final_model: Model = LinearPerSegmentModel(),
        cv: Union[None, int] = 3,
        features_to_use: Union[None, Literal[all], List[str]] = None,
        n_jobs: int = 1,
    ):
        """Init StackingEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble.
        final_model:
            Model which will be used to combine the base estimators.
        cv:
            Number of folds to use in the backtest. Backtest is not used for model evaluation but for prediction.
        features_to_use:
            Features except the forecasts of the base models to use in the 'final_model'.
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
        self.features_to_use = features_to_use
        self.n_jobs = n_jobs

    @staticmethod
    def _validate_pipeline_number(pipelines: List[Pipeline]):
        """Check that given valid number of pipelines."""
        if len(pipelines) < 2:
            raise ValueError("At least two pipelines are expected.")

    @staticmethod
    def _get_horizon(pipelines: List[Pipeline]) -> int:
        """Get ensemble's horizon."""
        horizons = list(set([pipeline.horizon for pipeline in pipelines]))
        if len(horizons) > 1:
            raise ValueError("All the pipelines should have the same horizon.")
        return horizons[0]

    @staticmethod
    def _validate_cv(cv: Optional[int]):
        """Check that given number of folds is grater than 1."""
        if cv is None:
            cv = 3
        elif cv < 2:
            raise ValueError("At least two folds for backtest are expected.")
        return cv

    def _validate_features_to_use(self, forecasts: List[TSDataset]):
        """Check that features from 'features_to_use' are available."""
        features_df = pd.concat([forecast.df for forecast in forecasts], axis=1)
        available_features = set(features_df.columns.get_level_values("feature")) - {"fold_number"}
        if isinstance(self.features_to_use, list):
            self.features_to_use = set(self.features_to_use)
            if len(self.features_to_use) == 0:
                self.features_to_use = None
            else:
                if not self.features_to_use.issubset(available_features):
                    unavailable_features = self.features_to_use - available_features
                    warnings.warn(f"Features {unavailable_features} are not found and will be dropped")
                    self.features_to_use = self.features_to_use.intersection(available_features)
        elif self.features_to_use == "all":
            self.features_to_use = available_features - {"target"}
        elif self.features_to_use is None:
            pass
        else:
            warnings.warn(
                "Feature list is passed in the wrong format."
                "Only the base models' forecasts will be used for the final forecast "
            )
            self.features_to_use = None

    @staticmethod
    def _fit_pipeline(pipeline: Pipeline, ts: TSDataset) -> Pipeline:
        """Fit given pipeline with ts."""
        tslogger.log(msg=f"Start fitting {pipeline.__repr__()}.")
        pipeline.fit(ts=ts)
        tslogger.log(msg=f"Pipeline {pipeline.__repr__()} is fitted.")
        return pipeline

    def _backtest_pipeline(self, pipeline: Pipeline, ts: TSDataset) -> TSDataset:
        """Get forecasts from backtest for given pipeline."""
        _, forecasts, _ = pipeline.backtest(ts, metrics=[MAE()], n_folds=self.cv)
        forecasts = TSDataset(df=forecasts, freq=ts.freq)
        return forecasts

    def _fit_final_model(self, ts: TSDataset):
        """Fit the 'final_model' on the forecasts of the base models."""
        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )
        self._validate_features_to_use(forecasts)

        features = self._make_features(ts=ts, forecasts=forecasts, train=True)
        self.final_model.fit(features)

    def fit(self, ts: TSDataset) -> "StackingEnsemble":
        """Fit pipelines in ensemble.

        Parameters
        ----------
        ts:
            TSDataset to fit ensemble.

        Returns
        -------
        StackingEnsemble:
            Fitted ensemble.
        """
        self._fit_final_model(ts)
        self.pipelines = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._fit_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )
        return self

    @staticmethod
    def _stack_targets(forecasts: List[TSDataset]) -> pd.DataFrame:
        """Stack targets from the forecasts."""
        targets = [
            forecast[:, :, "target"].rename({"target": f"regressor_target_{i}"}, axis=1)
            for i, forecast in enumerate(forecasts)
        ]
        targets = pd.concat(targets, axis=1)
        return targets

    def _get_features(self, forecasts: List[TSDataset]) -> pd.DataFrame:
        """Get features from the forecasts."""
        features_df = None
        features_left = self.features_to_use.copy()
        for forecast in forecasts:
            features_in_forecast = set(forecast.columns.get_level_values("feature"))
            features_new = features_left.intersection(features_in_forecast)
            if len(features_new) != 0:
                if features_df is None:
                    features_df = forecast[:, :, features_new]
                else:
                    features_df = pd.concat([features_df, forecast[:, :, features_new]], axis=1)
                features_left -= features_new
                if len(features_left) == 0:
                    break
        return features_df

    def _make_features(self, ts: TSDataset, forecasts: List[TSDataset], train: bool = False) -> TSDataset:
        """Prepare features for the 'final_model'."""
        features_df = self._stack_targets(forecasts=forecasts)
        if self.features_to_use is not None:
            features = self._get_features(forecasts=forecasts)
            features_df = pd.concat([features_df, features], axis=1)

        if train:
            targets_df = ts[forecasts[0].index.min() : forecasts[0].index.max(), :, "target"]
            ind = pd.date_range(start=forecasts[0].index.max(), periods=2, closed="right")
            new_index = features_df.index.append(ind)
            features_df = features_df.reindex(new_index)
            features_df.index.name = "timestamp"
            features_df.loc[ind, pd.IndexSlice[:, :]] = features_df.loc[
                forecasts[0].index.max(), pd.IndexSlice[:, :]
            ].values
        else:
            targets_df = forecasts[0][:, :, "target"]
            targets_df.loc[:] = np.NAN

        features_ts = TSDataset(df=targets_df, freq=forecasts[0].freq, df_exog=features_df)
        return features_ts

    @staticmethod
    def _forecast_pipeline(pipeline: Pipeline) -> TSDataset:
        """Make forecast with given pipeline."""
        tslogger.log(msg=f"Start forecasting with {pipeline.__repr__()}.")
        forecast = pipeline.forecast()
        tslogger.log(msg=f"Forecast is done with {pipeline.__repr__()}.")
        return forecast

    def forecast(self) -> TSDataset:
        """Forecast with ensemble: compute the combination of pipelines' forecasts using 'final_model'.

        Returns
        -------
        TSDataset:
            Dataset with forecasts.
        """
        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines
        )
        future = self._make_features(ts=None, forecasts=forecasts, train=False)
        forecast = self.final_model.forecast(future)
        return forecast
