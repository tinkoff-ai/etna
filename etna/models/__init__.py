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
>>> ts = TSDataset(TSDataset.to_dataset(df), "D")
>>> lag_transform = LagTransform(in_column="target", lags=[3, 4, 5])
>>> ts.fit_transform(transforms=[lag_transform])
>>> future_ts = ts.make_future(3)
>>> model = LinearPerSegmentModel()
>>> model.fit(ts)
LinearPerSegmentModel(fit_intercept = True, normalize = False, )
>>> forecast_ts = model.forecast(future_ts)
segment                 segment_0  ... segment_1
feature    regressor_target_lag_3  ...    target
timestamp                          ...
2021-04-11              -0.090673  ...  0.286764
2021-04-12              -0.665337  ...  0.295589
2021-04-13               0.365363  ...  0.374554
[3 rows x 8 columns]

There is a key note to mention: :code:`future_ts` and :code:`forecast_ts` are the same objects.
Method :code:`forecast` only fills 'target' columns in :code:`future_ts` and return reference to it.

>>> forecast_ts is future_ts
True
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
