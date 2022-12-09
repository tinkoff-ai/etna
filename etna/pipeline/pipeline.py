from typing import Sequence
from typing import cast

from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models.base import ContextIgnorantModelType
from etna.models.base import ContextRequiredModelType
from etna.models.base import ModelType
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline.base import BasePipeline
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.pipeline.mixins import SaveModelPipelineMixin
from etna.transforms.base import Transform


class Pipeline(ModelPipelinePredictMixin, SaveModelPipelineMixin, BasePipeline):
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
