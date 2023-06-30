from typing import Optional
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
from etna.pipeline.mixins import ModelPipelineParamsToTuneMixin
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.pipeline.mixins import SaveModelPipelineMixin
from etna.transforms.base import Transform


class Pipeline(ModelPipelinePredictMixin, ModelPipelineParamsToTuneMixin, SaveModelPipelineMixin, BasePipeline):
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
        self.ts.inverse_transform(self.transforms)
        return self

    def _forecast(self, ts: TSDataset, return_components: bool) -> TSDataset:
        """Make predictions."""
        if isinstance(self.model, get_args(ContextRequiredModelType)):
            self.model = cast(ContextRequiredModelType, self.model)
            future = ts.make_future(
                future_steps=self.horizon, transforms=self.transforms, tail_steps=self.model.context_size
            )
            predictions = self.model.forecast(
                ts=future, prediction_size=self.horizon, return_components=return_components
            )
        else:
            self.model = cast(ContextIgnorantModelType, self.model)
            future = ts.make_future(future_steps=self.horizon, transforms=self.transforms)
            predictions = self.model.forecast(ts=future, return_components=return_components)
        return predictions

    def forecast(
        self,
        ts: Optional[TSDataset] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        n_folds: int = 3,
        return_components: bool = False,
    ) -> TSDataset:
        """Make a forecast of the next points of a dataset.

        The result of forecasting starts from the last point of ``ts``, not including it.

        Parameters
        ----------
        ts:
            Dataset to forecast. If not given, dataset given during :py:meth:``fit`` is used.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions
        """
        if ts is None:
            if self.ts is None:
                raise ValueError(
                    "There is no ts to forecast! Pass ts into forecast method or make sure that pipeline is loaded with ts."
                )
            ts = self.ts

        self._validate_quantiles(quantiles=quantiles)
        self._validate_backtest_n_folds(n_folds=n_folds)

        if prediction_interval and isinstance(self.model, PredictionIntervalContextIgnorantAbstractModel):
            future = ts.make_future(future_steps=self.horizon, transforms=self.transforms)
            predictions = self.model.forecast(
                ts=future,
                prediction_interval=prediction_interval,
                quantiles=quantiles,
                return_components=return_components,
            )
        elif prediction_interval and isinstance(self.model, PredictionIntervalContextRequiredAbstractModel):
            future = ts.make_future(
                future_steps=self.horizon, transforms=self.transforms, tail_steps=self.model.context_size
            )
            predictions = self.model.forecast(
                ts=future,
                prediction_size=self.horizon,
                prediction_interval=prediction_interval,
                quantiles=quantiles,
                return_components=return_components,
            )
        else:
            predictions = super().forecast(
                ts=ts,
                prediction_interval=prediction_interval,
                quantiles=quantiles,
                n_folds=n_folds,
                return_components=return_components,
            )
        predictions.inverse_transform(self.transforms)
        return predictions
