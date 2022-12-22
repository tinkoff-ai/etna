from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed

from etna.datasets import TSDataset
from etna.ensembles.mixins import EnsembleMixin
from etna.ensembles.mixins import SaveEnsembleMixin
from etna.pipeline.base import BasePipeline


class DirectEnsemble(EnsembleMixin, SaveEnsembleMixin, BasePipeline):
    """DirectEnsemble is a pipeline that forecasts future values merging the forecasts of base pipelines.

    Ensemble expects several pipelines during init. These pipelines are expected to have different forecasting horizons.
    For each point in the future, forecast of the ensemble is forecast of base pipeline with the shortest horizon,
    which covers this point.

    Examples
    --------
    >>> from etna.datasets import generate_ar_df
    >>> from etna.datasets import TSDataset
    >>> from etna.ensembles import DirectEnsemble
    >>> from etna.models import NaiveModel
    >>> from etna.models import ProphetModel
    >>> from etna.pipeline import Pipeline
    >>> df = generate_ar_df(periods=30, start_time="2021-06-01", ar_coef=[1.2], n_segments=3)
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df_ts_format, "D")
    >>> prophet_pipeline = Pipeline(model=ProphetModel(), transforms=[], horizon=3)
    >>> naive_pipeline = Pipeline(model=NaiveModel(lag=10), transforms=[], horizon=5)
    >>> ensemble = DirectEnsemble(pipelines=[prophet_pipeline, naive_pipeline])
    >>> _ = ensemble.fit(ts=ts)
    >>> forecast = ensemble.forecast()
    >>> forecast
    segment    segment_0 segment_1 segment_2
    feature       target    target    target
    timestamp
    2021-07-01    -10.37   -232.60    163.16
    2021-07-02    -10.59   -242.05    169.62
    2021-07-03    -11.41   -253.82    177.62
    2021-07-04     -5.85   -139.57     96.99
    2021-07-05     -6.11   -167.69    116.59
    """

    def __init__(
        self,
        pipelines: List[BasePipeline],
        n_jobs: int = 1,
        joblib_params: Optional[Dict[str, Any]] = None,
    ):
        """Init DirectEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`

        Raises
        ------
        ValueError:
            If two or more pipelines have the same horizons.
        """
        self._validate_pipeline_number(pipelines=pipelines)
        self.pipelines = pipelines
        self.n_jobs = n_jobs
        if joblib_params is None:
            self.joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")
        else:
            self.joblib_params = joblib_params
        super().__init__(horizon=self._get_horizon(pipelines=pipelines))

    @staticmethod
    def _get_horizon(pipelines: List[BasePipeline]) -> int:
        """Get ensemble's horizon."""
        horizons = {pipeline.horizon for pipeline in pipelines}
        if len(horizons) != len(pipelines):
            raise ValueError("All the pipelines should have pairwise different horizons.")
        return max(horizons)

    def fit(self, ts: TSDataset) -> "DirectEnsemble":
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
        return self

    def _merge(self, forecasts: List[TSDataset]) -> TSDataset:
        """Merge the forecasts of base pipelines according to the direct strategy."""
        segments = sorted(forecasts[0].segments)
        horizons = [pipeline.horizon for pipeline in self.pipelines]
        pipelines_order = np.argsort(horizons)[::-1]
        # TODO: Fix slicing with explicit passing the segments in issue #775
        forecast_df = forecasts[pipelines_order[0]][:, segments, "target"]
        for idx in pipelines_order:
            # TODO: Fix slicing with explicit passing the segments in issue #775
            horizon, forecast = horizons[idx], forecasts[idx][:, segments, "target"]
            forecast_df.iloc[:horizon] = forecast
        forecast_dataset = TSDataset(df=forecast_df, freq=forecasts[0].freq)
        return forecast_dataset

    def _forecast(self) -> TSDataset:
        """Make predictions.

        In each point in the future, forecast of the ensemble is forecast of base pipeline with the shortest horizon,
        which covers this point.
        """
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")

        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines
        )
        forecast = self._merge(forecasts=forecasts)
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

        horizons = [pipeline.horizon for pipeline in self.pipelines]
        pipeline = self.pipelines[np.argmin(horizons)]
        prediction = self._predict_pipeline(
            ts=ts, pipeline=pipeline, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )
        return prediction
