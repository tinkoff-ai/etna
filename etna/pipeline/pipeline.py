from typing import Iterable

from etna.datasets import TSDataset
from etna.models.base import Model
from etna.transforms.base import Transform


class Pipeline:
    """Pipeline of transforms with a final estimator."""

    def __init__(self, model: Model, transforms: Iterable[Transform] = (), horizon: int = 1):
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
        self.horizon = horizon
        self.ts = None

    def fit(self, ts: TSDataset) -> "Pipeline":
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
        self.ts = ts
        self.ts.fit_transform(self.transforms)
        self.model.fit(self.ts)
        return self

    def forecast(self) -> TSDataset:
        """ Make predictions.

        Returns
        -------
        TSDataset
            TSDataset with forecast
        """
        future = self.ts.make_future(self.horizon)
        predictions = self.model.forecast(future)
        return predictions
