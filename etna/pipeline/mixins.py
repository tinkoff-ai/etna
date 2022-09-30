from copy import deepcopy
from typing import Optional
from typing import Sequence
from typing import cast

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

    ts: Optional[TSDataset]
    model: ModelType
    transforms: Sequence[Transform]

    def _recreate_ts(self, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> TSDataset:
        """Recreate ``self.ts`` on given timestamp range in state before any transformations."""
        self.ts = cast(TSDataset, self.ts)
        df = self.ts.raw_df.copy()
        # we make it through deepcopy to handle df_exog=None
        df_exog = deepcopy(self.ts.df_exog)
        freq = self.ts.freq
        known_future = self.ts.known_future

        # correct start_timestamp taking into account context size
        timestamp_indices = pd.Series(np.arange(len(df.index)), index=df.index)
        start_idx = timestamp_indices[start_timestamp]
        start_idx = max(0, start_idx - self.model.context_size)
        start_timestamp = timestamp_indices.index[start_idx]

        df = df[start_timestamp:end_timestamp]
        ts = TSDataset(df=df, df_exog=df_exog, freq=freq, known_future=known_future)
        return ts

    def _determine_prediction_size(self, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp) -> int:
        self.ts = cast(TSDataset, self.ts)
        timestamp_indices = pd.Series(np.arange(len(self.ts.index)), index=self.ts.index)
        timestamps = timestamp_indices[start_timestamp:end_timestamp]
        return len(timestamps)

    def _predict(
        self,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> TSDataset:
        self.ts = cast(TSDataset, self.ts)
        ts = self._recreate_ts(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
        prediction_size = self._determine_prediction_size(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
        ts.transform(transforms=self.transforms)

        if prediction_interval and isinstance(self.model, get_args(NonPredictionIntervalModelType)):
            raise NotImplementedError(f"Model {self.model.__class__.__name__} doesn't support prediction intervals!")

        if isinstance(self.model, NonPredictionIntervalContextIgnorantAbstractModel):
            results = self.model.predict(ts=ts)
        elif isinstance(self.model, NonPredictionIntervalContextRequiredAbstractModel):
            results = self.model.predict(ts=ts, prediction_size=prediction_size)
        elif isinstance(self.model, PredictionIntervalContextIgnorantAbstractModel):
            results = self.model.predict(ts=ts, prediction_interval=prediction_interval, quantiles=quantiles)
        elif isinstance(self.model, PredictionIntervalContextRequiredAbstractModel):
            results = self.model.predict(
                ts=ts, prediction_size=prediction_size, prediction_interval=prediction_interval, quantiles=quantiles
            )
        else:
            raise NotImplementedError(f"Unknown model type: {self.model.__class__.__name__}!")
        return results
