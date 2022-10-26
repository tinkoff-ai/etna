from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from typing import cast

import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.ensemble import RandomForestRegressor
from typing_extensions import Literal

from etna.analysis.feature_relevance.relevance_table import TreeBasedRegressor
from etna.datasets import TSDataset
from etna.ensembles import EnsembleMixin
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.pipeline.base import BasePipeline


class VotingEnsemble(BasePipeline, EnsembleMixin):
    """VotingEnsemble is a pipeline that forecast future values with weighted averaging of it's pipelines forecasts.

    Examples
    --------
    >>> from etna.datasets import generate_ar_df
    >>> from etna.datasets import TSDataset
    >>> from etna.ensembles import VotingEnsemble
    >>> from etna.models import NaiveModel
    >>> from etna.models import ProphetModel
    >>> from etna.pipeline import Pipeline
    >>> df = generate_ar_df(periods=30, start_time="2021-06-01", ar_coef=[1.2], n_segments=3)
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df_ts_format, "D")
    >>> prophet_pipeline = Pipeline(model=ProphetModel(), transforms=[], horizon=7)
    >>> naive_pipeline = Pipeline(model=NaiveModel(lag=10), transforms=[], horizon=7)
    >>> ensemble = VotingEnsemble(
    ...     pipelines=[prophet_pipeline, naive_pipeline],
    ...     weights=[0.7, 0.3]
    ... )
    >>> _ = ensemble.fit(ts=ts)
    >>> forecast = ensemble.forecast()
    >>> forecast
    segment         segment_0        segment_1       segment_2
    feature	       target           target	        target
    timestamp
    2021-07-01	        -8.84	       -186.67	        130.99
    2021-07-02	        -8.96	       -198.16	        138.81
    2021-07-03	        -9.57	       -212.48	        148.48
    2021-07-04	       -10.48	       -229.16	        160.13
    2021-07-05	       -11.20          -248.93	        174.39
    2021-07-06	       -12.47	       -281.90	        197.82
    2021-07-07	       -13.51	       -307.02	        215.73
    """

    def __init__(
        self,
        pipelines: List[BasePipeline],
        weights: Optional[Union[List[float], Literal["auto"]]] = None,
        regressor: Optional[TreeBasedRegressor] = None,
        n_folds: int = 3,
        n_jobs: int = 1,
        joblib_params: Optional[Dict[str, Any]] = None,
    ):
        """Init VotingEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble
        weights:
            List of pipelines' weights.

            * If None, use uniform weights

            * If List[float], use this weights for the base estimators, weights will be normalized automatically

            * If "auto", use importances of the base estimators forecasts as weights of base estimators

        regressor:
            Regression model with fit/predict interface which will be used to evaluate weights of the base estimators.
            It should have ``feature_importances_`` property (e.g. all tree-based regressors in sklearn)
        n_folds:
            Number of folds to use in the backtest.
            Backtest is used to obtain the forecasts from the base estimators;
            forecasts will be used to evaluate the estimator's weights.
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`

        Raises
        ------
        ValueError:
            If the number of the pipelines is less than 2 or pipelines have different horizons.
        """
        self._validate_pipeline_number(pipelines=pipelines)
        self._validate_weights(weights=weights, pipelines_number=len(pipelines))
        self._validate_backtest_n_folds(n_folds)
        self.weights = weights
        self.processed_weights: Optional[List[float]] = None
        self.regressor = RandomForestRegressor(n_estimators=5) if regressor is None else regressor
        self.n_folds = n_folds
        self.pipelines = pipelines
        self.n_jobs = n_jobs
        if joblib_params is None:
            self.joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")
        else:
            self.joblib_params = joblib_params
        super().__init__(horizon=self._get_horizon(pipelines=pipelines))

    @staticmethod
    def _validate_weights(weights: Optional[Union[List[float], Literal["auto"]]], pipelines_number: int):
        """Validate the format of weights parameter."""
        if weights is None or weights == "auto":
            pass
        elif isinstance(weights, list):
            if len(weights) != pipelines_number:
                raise ValueError("Weights size should be equal to pipelines number.")
        else:
            raise ValueError("Invalid format of weights is passed!")

    def _backtest_pipeline(self, pipeline: BasePipeline, ts: TSDataset) -> TSDataset:
        """Get forecasts from backtest for given pipeline."""
        with tslogger.disable():
            _, forecasts, _ = pipeline.backtest(ts, metrics=[MAE()], n_folds=self.n_folds)
        forecasts = TSDataset(df=forecasts, freq=ts.freq)
        return forecasts

    def _process_weights(self) -> List[float]:
        """Get the weights of base estimators depending on the weights mode."""
        if self.weights is None:
            weights = [1.0 for _ in range(len(self.pipelines))]
        elif self.weights == "auto":
            if self.ts is None:
                raise ValueError("Something went wrong, ts is None!")

            forecasts = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
                delayed(self._backtest_pipeline)(pipeline=pipeline, ts=deepcopy(self.ts)) for pipeline in self.pipelines
            )

            x = pd.concat(
                [
                    forecast[:, :, "target"].rename({"target": f"target_{i}"}, axis=1)
                    for i, forecast in enumerate(forecasts)
                ],
                axis=1,
            )
            x = pd.concat([x.loc[:, segment] for segment in self.ts.segments], axis=0)

            y = pd.concat(
                [
                    self.ts[forecasts[0].index.min() : forecasts[0].index.max(), segment, "target"]
                    for segment in self.ts.segments
                ],
                axis=0,
            )

            self.regressor.fit(x, y)
            weights = self.regressor.feature_importances_
        else:
            weights = self.weights
        common_weight = sum(weights)
        weights = [w / common_weight for w in weights]
        return weights

    def fit(self, ts: TSDataset) -> "VotingEnsemble":
        """Fit pipelines in ensemble.

        Parameters
        ----------
        ts:
            TSDataset to fit ensemble

        Returns
        -------
        self:
            Fitted ensemble
        """
        self.ts = ts
        self.pipelines = Parallel(n_jobs=self.n_jobs, **self.joblib_params)(
            delayed(self._fit_pipeline)(pipeline=pipeline, ts=deepcopy(ts)) for pipeline in self.pipelines
        )
        self.processed_weights = self._process_weights()
        return self

    def _vote(self, forecasts: List[TSDataset]) -> TSDataset:
        """Get average forecast."""
        if self.processed_weights is None:
            raise ValueError("Ensemble is not fitted! Fit the ensemble before calling the forecast!")

        forecast_df = sum(
            [forecast[:, :, "target"] * weight for forecast, weight in zip(forecasts, self.processed_weights)]
        )
        forecast_dataset = TSDataset(df=forecast_df, freq=forecasts[0].freq)
        return forecast_dataset

    def _forecast(self) -> TSDataset:
        """Make predictions.

        Compute weighted average of pipelines' forecasts
        """
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")

        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines
        )
        forecast = self._vote(forecasts=forecasts)
        return forecast

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> TSDataset:
        if prediction_interval:
            raise NotImplementedError(f"Ensemble {self.__class__.__name__} doesn't support prediction intervals!")

        self.ts = cast(TSDataset, self.ts)
        predictions = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._predict_pipeline)(
                ts=ts, pipeline=pipeline, start_timestamp=start_timestamp, end_timestamp=end_timestamp
            )
            for pipeline in self.pipelines
        )
        predictions = self._vote(forecasts=predictions)
        return predictions
