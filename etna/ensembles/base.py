from typing import List

from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.pipeline.base import BasePipeline


class EnsembleMixin:
    """Base mixin for the ensembles."""

    @staticmethod
    def _validate_pipeline_number(pipelines: List[BasePipeline]):
        """Check that given valid number of pipelines."""
        if len(pipelines) < 2:
            raise ValueError("At least two pipelines are expected.")

    @staticmethod
    def _get_horizon(pipelines: List[BasePipeline]) -> int:
        """Get ensemble's horizon."""
        horizons = set([pipeline.horizon for pipeline in pipelines])
        if len(horizons) > 1:
            raise ValueError("All the pipelines should have the same horizon.")
        return horizons.pop()

    @staticmethod
    def _fit_pipeline(pipeline: BasePipeline, ts: TSDataset) -> BasePipeline:
        """Fit given pipeline with ts."""
        tslogger.log(msg=f"Start fitting {pipeline}.")
        pipeline.fit(ts=ts)
        tslogger.log(msg=f"Pipeline {pipeline} is fitted.")
        return pipeline

    @staticmethod
    def _forecast_pipeline(pipeline: BasePipeline) -> TSDataset:
        """Make forecast with given pipeline."""
        tslogger.log(msg=f"Start forecasting with {pipeline}.")
        forecast = pipeline.forecast()
        tslogger.log(msg=f"Forecast is done with {pipeline}.")
        return forecast
