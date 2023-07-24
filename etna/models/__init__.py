"""
Module with models for time-series forecasting.

Basic usage
-----------

Models are used to make predictions. Let's look at the basic example of usage:

>>> import pandas as pd
>>> from etna.datasets import TSDataset, generate_ar_df
>>> from etna.transforms import LagTransform
>>> from etna.models import LinearPerSegmentModel
>>>
>>> df = generate_ar_df(periods=100, start_time="2021-01-01", ar_coef=[1/2], n_segments=2)
>>> ts = TSDataset(TSDataset.to_dataset(df), freq="D")
>>> lag_transform = LagTransform(in_column="target", lags=[3, 4, 5])
>>> ts.fit_transform(transforms=[lag_transform])
>>> future_ts = ts.make_future(future_steps=3, transforms=[lag_transform])
>>> model = LinearPerSegmentModel()
>>> model.fit(ts)
LinearPerSegmentModel(fit_intercept = True, )
>>> forecast_ts = model.forecast(future_ts)
>>> forecast_ts is future_ts
True

There is a key note to mention: :code:`future_ts` and :code:`forecast_ts` are the same objects.
Method :code:`forecast` only fills 'target' columns in :code:`future_ts` and return reference to it.
"""

from etna import SETTINGS
from etna.models.autoarima import AutoARIMAModel
from etna.models.base import BaseAdapter
from etna.models.base import ContextIgnorantModelType
from etna.models.base import ContextRequiredModelType
from etna.models.base import ModelType
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import NonPredictionIntervalModelType
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalModelType
from etna.models.catboost import CatBoostMultiSegmentModel
from etna.models.catboost import CatBoostPerSegmentModel
from etna.models.deadline_ma import DeadlineMovingAverageModel
from etna.models.holt_winters import HoltModel
from etna.models.holt_winters import HoltWintersModel
from etna.models.holt_winters import SimpleExpSmoothingModel
from etna.models.linear import ElasticMultiSegmentModel
from etna.models.linear import ElasticPerSegmentModel
from etna.models.linear import LinearMultiSegmentModel
from etna.models.linear import LinearPerSegmentModel
from etna.models.moving_average import MovingAverageModel
from etna.models.naive import NaiveModel
from etna.models.sarimax import SARIMAXModel
from etna.models.seasonal_ma import SeasonalMovingAverageModel
from etna.models.sklearn import SklearnMultiSegmentModel
from etna.models.sklearn import SklearnPerSegmentModel
from etna.models.tbats import BATSModel
from etna.models.tbats import TBATSModel

if SETTINGS.prophet_required:
    from etna.models.prophet import ProphetModel

if SETTINGS.statsforecast_required:
    from etna.models.statsforecast import StatsForecastARIMAModel
    from etna.models.statsforecast import StatsForecastAutoARIMAModel
    from etna.models.statsforecast import StatsForecastAutoCESModel
    from etna.models.statsforecast import StatsForecastAutoETSModel
    from etna.models.statsforecast import StatsForecastAutoThetaModel
