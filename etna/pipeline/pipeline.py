from typing import Optional
from typing import Sequence
from typing import cast

import pandas as pd
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models.base import ContextIgnorantModelType
from etna.models.base import ContextRequiredModelType
from etna.models.base import ModelType
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline.base import BasePipeline
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.transforms.base import Transform


class Pipeline(BasePipeline, ModelPipelinePredictMixin):
    """Pipeline of transforms with a final estimator."""

    def __init__(self, model: ModelType, transforms: Sequence[Transform] = (), horizon: int = 1):
        """
        Create instance of Pipeline with given parameters.

        Parameters
        ----------
        model:
            Instance of the etna Model
        transforms:
            Sequence of the transforms
        horizon:
            Number of timestamps in the future for forecasting
        """
        self.model = model
        self.transforms = transforms
        super().__init__(horizon=horizon)

    def fit(self, ts: TSDataset) -> "Pipeline":
        """Fit the Pipeline.

        Fit and apply given transforms to the data, then fit the model on the transformed data.

        Parameters
        ----------
        ts:
            Dataset with timeseries data

        Returns
        -------
        :
            Fitted Pipeline instance
        """
        self.ts = ts
        self.ts.fit_transform(self.transforms)
        self.model.fit(self.ts)
        self.ts.inverse_transform()
        return self

    def _forecast(self) -> TSDataset:
        """Make predictions."""
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")

        if isinstance(self.model, get_args(ContextRequiredModelType)):
            self.model = cast(ContextRequiredModelType, self.model)
            future = self.ts.make_future(future_steps=self.horizon, tail_steps=self.model.context_size)
            predictions = self.model.forecast(ts=future, prediction_size=self.horizon)
        else:
            self.model = cast(ContextIgnorantModelType, self.model)
            future = self.ts.make_future(future_steps=self.horizon)
            predictions = self.model.forecast(ts=future)
        return predictions

    def forecast(
        self, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975), n_folds: int = 3
    ) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation

        Returns
        -------
        :
            Dataset with predictions
        """
        if self.ts is None:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted! Fit the {self.__class__.__name__} "
                f"before calling forecast method."
            )
        self._validate_quantiles(quantiles=quantiles)
        self._validate_backtest_n_folds(n_folds=n_folds)

        if prediction_interval and isinstance(self.model, PredictionIntervalContextIgnorantAbstractModel):
            future = self.ts.make_future(future_steps=self.horizon)
            predictions = self.model.forecast(ts=future, prediction_interval=prediction_interval, quantiles=quantiles)
        elif prediction_interval and isinstance(self.model, PredictionIntervalContextRequiredAbstractModel):
            future = self.ts.make_future(future_steps=self.horizon, tail_steps=self.model.context_size)
            predictions = self.model.forecast(
                ts=future, prediction_size=self.horizon, prediction_interval=prediction_interval, quantiles=quantiles
            )
        else:
            predictions = super().forecast(
                prediction_interval=prediction_interval, quantiles=quantiles, n_folds=n_folds
            )
        return predictions

    def predict(
        self,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
    ) -> TSDataset:
        """Make in-sample predictions in a given range.

        Currently, in situation when segments start with different timestamps
        we only guarantee to work with ``start_timestamp`` >= beginning of all segments.

        Parameters
        ----------
        start_timestamp:
            First timestamp of prediction range to return, should be >= than first timestamp in ``self.ts``;
            expected that beginning of each segment <= ``start_timestamp``;
            if isn't set the first timestamp where each segment began is taken.
        end_timestamp:
            Last timestamp of prediction range to return; if isn't set the last timestamp of ``self.ts`` is taken.
            Expected that value is less or equal to the last timestamp in ``self.ts``.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.

        Returns
        -------
        :
            Dataset with predictions in ``[start_timestamp, end_timestamp]`` range.

        Raises
        ------
        ValueError:
            Pipeline wasn't fitted.
        ValueError:
            Value of ``end_timestamp`` is less than ``start_timestamp``.
        ValueError:
            Value of ``start_timestamp`` goes before point where each segment started.
        ValueError:
            Value of ``end_timestamp`` goes after the last timestamp.
        """
        if self.ts is None:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted! Fit the {self.__class__.__name__} "
                f"before calling predict method."
            )

        start_timestamp, end_timestamp = self._validate_predict_timestamps(
            ts=self.ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )
        self._validate_quantiles(quantiles=quantiles)
        result = self._predict(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
        )
        return result
