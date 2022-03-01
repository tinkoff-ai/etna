from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from joblib import Parallel
from joblib import delayed

from etna.datasets import TSDataset
from etna.ensembles import EnsembleMixin
from etna.pipeline import BasePipeline


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
        weights: Optional[List[float]] = None,
        n_jobs: int = 1,
        joblib_params: Dict[str, Any] = dict(verbose=11, backend="multiprocessing", mmap_mode="c"),
    ):
        """Init VotingEnsemble.

        Parameters
        ----------
        pipelines:
            List of pipelines that should be used in ensemble
        weights:
            List of pipelines' weights; weights will be normalized automatically.
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for joblib.Parallel

        Raises
        ------
        ValueError:
            If the number of the pipelines is less than 2 or pipelines have different horizons.
        """
        self._validate_pipeline_number(pipelines=pipelines)
        self.weights = self._process_weights(weights=weights, pipelines_number=len(pipelines))
        self.pipelines = pipelines
        self.n_jobs = n_jobs
        self.joblib_params = joblib_params
        super().__init__(horizon=self._get_horizon(pipelines=pipelines))

    @staticmethod
    def _process_weights(weights: Optional[List[float]], pipelines_number: int) -> List[float]:
        """Process weights: if weights are not given, set them with default values, normalize weights."""
        if weights is None:
            weights = [1 / pipelines_number for _ in range(pipelines_number)]
        elif len(weights) != pipelines_number:
            raise ValueError("Weights size should be equal to pipelines number.")
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
        return self

    def _vote(self, forecasts: List[TSDataset]) -> TSDataset:
        """Get average forecast."""
        forecast_df = sum([forecast[:, :, "target"] * weight for forecast, weight in zip(forecasts, self.weights)])
        forecast_dataset = TSDataset(df=forecast_df, freq=forecasts[0].freq)
        return forecast_dataset

    def _forecast(self) -> TSDataset:
        """Make predictions.
        Compute weighted average of pipelines' forecasts
        """
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None inside the _forecast!")

        forecasts = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=11)(
            delayed(self._forecast_pipeline)(pipeline=pipeline) for pipeline in self.pipelines
        )
        forecast = self._vote(forecasts=forecasts)
        return forecast
