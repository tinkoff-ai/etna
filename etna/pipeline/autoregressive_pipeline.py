from copy import deepcopy
from typing import Iterable

import pandas as pd

from etna.datasets import TSDataset
from etna.models.base import Model
from etna.transforms.base import Transform
from etna.pipeline import Pipeline


class AutoRegressivePipeline(Pipeline):
    """Pipeline that make regressive models autoregressive.

    TODO: Add example
    """

    def __init__(self, model: Model, horizon: int, transforms: Iterable[Transform] = (), step: int = 1):
        """
        Create instance of AutoRegressivePipeline with given parameters.

        Parameters
        ----------
        model:
            Instance of the etna Model
        horizon:
            Number of timestamps in the future for forecasting
        transforms:
            Sequence of the transforms
        step:
            Size of prediction for one step of forecasting
        """
        self.model = model
        self.horizon = horizon
        self.transforms = transforms
        self.step = step
        self.step_pipeline = Pipeline(model, transforms, step)

    def fit(self, ts: TSDataset) -> Pipeline:
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
        self.step_pipeline.fit(ts)
        return self

    @staticmethod
    def _update_pipeline_ts(pipeline: Pipeline, ts: TSDataset) -> None:
        """Append df to self.step_pipeline.ts."""
        pipeline.ts.df = pd.concat([
            pipeline.ts.df, ts.df
        ])

    def forecast(self) -> TSDataset:
        """Make predictions.

        Returns
        -------
        TSDataset
            TSDataset with forecast
        """
        step_forecast_pipeline = deepcopy(self.step_pipeline)
        to_forecast = self.horizon
        predictions_steps = []
        freq = None
        while to_forecast >= self.step:
            cur_step = min(self.step, to_forecast)
            predictions_step = step_forecast_pipeline.forecast()
            self._update_pipeline_ts(step_forecast_pipeline, predictions_step)
            predictions_steps.append(predictions_step.df.iloc[:cur_step])
            freq = predictions_step.freq
        predictions_df = pd.concat(predictions_steps)
        predictions_ts = TSDataset(predictions_df, freq=freq)
        return predictions_ts
