from typing import List
from typing import Optional

import pandas as pd

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
        horizons = {pipeline.horizon for pipeline in pipelines}
        if len(horizons) > 1:
            raise ValueError("All the pipelines should have the same horizon.")
        return horizons.pop()

    @staticmethod
    def _fit_pipeline(pipeline: BasePipeline, ts: TSDataset) -> BasePipeline:
        """Fit given pipeline with ``ts``."""
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

    @staticmethod
    def _predict_pipeline(
        ts: TSDataset,
        pipeline: BasePipeline,
        start_timestamp: Optional[pd.Timestamp],
        end_timestamp: Optional[pd.Timestamp],
    ) -> TSDataset:
        """Make predict with given pipeline."""
        tslogger.log(msg=f"Start prediction with {pipeline}.")
        prediction = pipeline.predict(ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
        tslogger.log(msg=f"Prediction is done with {pipeline}.")
        return prediction
