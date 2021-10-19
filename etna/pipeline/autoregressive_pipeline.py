import warnings
from copy import deepcopy
from typing import Iterable

import pandas as pd

from etna.datasets import TSDataset
from etna.models.base import Model
from etna.pipeline.pipeline import Pipeline
from etna.transforms import Transform


class AutoRegressivePipeline(Pipeline):
    """Pipeline that make regressive models autoregressive.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import LinearPerSegmentModel
    >>> from etna.transforms import LagTransform
    >>> classic_df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> df = TSDataset.to_dataset(df=classic_df)
    >>> ts = TSDataset(df, freq="D")
    >>> horizon = 7
    >>> transforms = [
    ...     LagTransform(in_column="target", lags=list(range(1, horizon+1)))
    ... ]
    >>> model = LinearPerSegmentModel()
    >>> pipeline = AutoRegressivePipeline(model, horizon, transforms, step=1)
    >>> _ = pipeline.fit(ts=ts)
    >>> forecast = pipeline.forecast()
    >>> forecast[:, :, "target"]
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12      0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
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
        self.transforms = transforms
        self.model = model

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
        self.ts = deepcopy(ts)
        ts.fit_transform(self.transforms)
        self.model.fit(ts)
        return self

    def forecast(self) -> TSDataset:
        """Make predictions.

        Returns
        -------
        TSDataset
            TSDataset with forecast
        """
        prediction_df = self.ts.to_pandas()
        to_forecast = self.horizon
        while to_forecast > 0:
            cur_step = min(self.step, to_forecast)
            cur_ts = TSDataset(prediction_df, freq=self.ts.freq)
            # manually set transforms in cur_ts, otherwise make_future won't know about them
            cur_ts.transforms = self.transforms
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    message="TSDataset freq can't be inferred",
                    action="ignore",
                )
                warnings.filterwarnings(
                    message="You probably set wrong freq.",
                    action="ignore",
                )
                cur_ts_forecast = cur_ts.make_future(cur_step)
            cur_ts_future = self.model.forecast(cur_ts_forecast)
            prediction_df = pd.concat([prediction_df, cur_ts_future.to_pandas()[prediction_df.columns]])
            to_forecast -= cur_step

        prediction_ts = TSDataset(prediction_df.tail(self.horizon), freq=self.ts.freq)
        # add all other features to forecast by making transform + inverse_transform
        prediction_ts.transform(self.transforms)
        prediction_ts.inverse_transform()
        return prediction_ts
