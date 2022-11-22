from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models import ModelType
from etna.models import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models import NonPredictionIntervalContextRequiredAbstractModel
from etna.models import NonPredictionIntervalModelType
from etna.models import PredictionIntervalContextIgnorantAbstractModel
from etna.models import PredictionIntervalContextRequiredAbstractModel
from etna.transforms import Transform


class ModelPipelinePredictMixin:
    """Mixin for pipelines with model inside with implementation of ``_predict`` method."""

    model: ModelType
    transforms: Sequence[Transform]

    def _create_ts(self, ts: TSDataset, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> TSDataset:
        """Create ``TSDataset`` to make predictions on."""
        df = deepcopy(ts.raw_df)
        df_exog = deepcopy(ts.df_exog)
        freq = deepcopy(ts.freq)
        known_future = deepcopy(ts.known_future)

        df_to_transform = df[:end_timestamp]
        cur_ts = TSDataset(df=df_to_transform, df_exog=df_exog, freq=freq, known_future=known_future)
        cur_ts.transform(transforms=self.transforms)

        # correct start_timestamp taking into account context size
        timestamp_indices = pd.Series(np.arange(len(df.index)), index=df.index)
        start_idx = timestamp_indices[start_timestamp]
        start_idx = max(0, start_idx - self.model.context_size)
        start_timestamp = timestamp_indices.index[start_idx]

        cur_ts.df = cur_ts.df[start_timestamp:end_timestamp]
        return cur_ts

    def _determine_prediction_size(
        self, ts: TSDataset, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp
    ) -> int:
        timestamp_indices = pd.Series(np.arange(len(ts.index)), index=ts.index)
        timestamps = timestamp_indices[start_timestamp:end_timestamp]
        return len(timestamps)

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> TSDataset:
        predict_ts = self._create_ts(ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)
        prediction_size = self._determine_prediction_size(
            ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )

        if prediction_interval and isinstance(self.model, get_args(NonPredictionIntervalModelType)):
            raise NotImplementedError(f"Model {self.model.__class__.__name__} doesn't support prediction intervals!")

        if isinstance(self.model, NonPredictionIntervalContextIgnorantAbstractModel):
            results = self.model.predict(ts=predict_ts)
        elif isinstance(self.model, NonPredictionIntervalContextRequiredAbstractModel):
            results = self.model.predict(ts=predict_ts, prediction_size=prediction_size)
        elif isinstance(self.model, PredictionIntervalContextIgnorantAbstractModel):
            results = self.model.predict(ts=predict_ts, prediction_interval=prediction_interval, quantiles=quantiles)
        elif isinstance(self.model, PredictionIntervalContextRequiredAbstractModel):
            results = self.model.predict(
                ts=predict_ts,
                prediction_size=prediction_size,
                prediction_interval=prediction_interval,
                quantiles=quantiles,
            )
        else:
            raise NotImplementedError(f"Unknown model type: {self.model.__class__.__name__}!")
        return results
