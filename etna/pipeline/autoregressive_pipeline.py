import warnings
from typing import Sequence
from typing import cast

import pandas as pd
from typing_extensions import get_args

from etna.datasets import TSDataset
from etna.models.base import ContextIgnorantModelType
from etna.models.base import ContextRequiredModelType
from etna.models.base import ModelType
from etna.pipeline.base import BasePipeline
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.transforms import Transform


class AutoRegressivePipeline(ModelPipelinePredictMixin, BasePipeline):
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
    >>> pd.options.display.float_format = '{:,.2f}'.format
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

    def __init__(self, model: ModelType, horizon: int, transforms: Sequence[Transform] = (), step: int = 1):
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
        self.transforms = transforms
        self.step = step
        super().__init__(horizon=horizon)

    def fit(self, ts: TSDataset) -> "AutoRegressivePipeline":
        """Fit the AutoRegressivePipeline.

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
        ts.fit_transform(self.transforms)
        self.model.fit(ts)
        self.ts.inverse_transform()
        return self

    def _create_predictions_template(self) -> pd.DataFrame:
        """Create dataframe to fill with forecasts."""
        if self.ts is None:
            raise ValueError(
                "AutoRegressivePipeline is not fitted! Fit the AutoRegressivePipeline before calling forecast method."
            )
        prediction_df = self.ts[:, :, "target"]
        future_dates = pd.date_range(
            start=prediction_df.index.max(), periods=self.horizon + 1, freq=self.ts.freq, closed="right"
        )
        prediction_df = prediction_df.reindex(prediction_df.index.append(future_dates))
        prediction_df.index.name = "timestamp"
        return prediction_df

    def _forecast(self) -> TSDataset:
        """Make predictions."""
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")
        prediction_df = self._create_predictions_template()

        for idx_start in range(0, self.horizon, self.step):
            current_step = min(self.step, self.horizon - idx_start)
            current_idx_border = self.ts.index.shape[0] + idx_start
            current_ts = TSDataset(
                df=prediction_df.iloc[:current_idx_border],
                freq=self.ts.freq,
                df_exog=self.ts.df_exog,
                known_future=self.ts.known_future,
            )
            # manually set transforms in current_ts, otherwise make_future won't know about them
            current_ts.transforms = self.transforms

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    message="TSDataset freq can't be inferred",
                    action="ignore",
                )
                warnings.filterwarnings(
                    message="You probably set wrong freq.",
                    action="ignore",
                )

                if isinstance(self.model, get_args(ContextRequiredModelType)):
                    self.model = cast(ContextRequiredModelType, self.model)
                    current_ts_forecast = current_ts.make_future(
                        future_steps=current_step, tail_steps=self.model.context_size
                    )
                    current_ts_future = self.model.forecast(ts=current_ts_forecast, prediction_size=current_step)
                else:
                    self.model = cast(ContextIgnorantModelType, self.model)
                    current_ts_forecast = current_ts.make_future(future_steps=current_step)
                    current_ts_future = self.model.forecast(ts=current_ts_forecast)

            prediction_df = prediction_df.combine_first(current_ts_future.to_pandas()[prediction_df.columns])

        # construct dataset and add all features
        prediction_ts = TSDataset(
            df=prediction_df, freq=self.ts.freq, df_exog=self.ts.df_exog, known_future=self.ts.known_future
        )
        prediction_ts.transform(self.transforms)
        prediction_ts.inverse_transform()
        # cut only last timestamps from result dataset
        prediction_ts.df = prediction_ts.df.tail(self.horizon)
        prediction_ts.raw_df = prediction_ts.raw_df.tail(self.horizon)
        return prediction_ts
