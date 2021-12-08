from etna import SETTINGS
from etna.models.base import Model
from etna.models.base import PerSegmentModel
from etna.models.catboost import CatBoostModelMultiSegment
from etna.models.catboost import CatBoostModelPerSegment
from etna.models.linear import ElasticMultiSegmentModel
from etna.models.linear import ElasticPerSegmentModel
from etna.models.linear import LinearMultiSegmentModel
from etna.models.linear import LinearPerSegmentModel
from etna.models.moving_average import MovingAverageModel
from etna.models.naive import NaiveModel
from etna.models.sarimax import SARIMAXModel
from etna.models.seasonal_ma import SeasonalMovingAverageModel

if SETTINGS.prophet_required:
    from etna.models.prophet import ProphetModel
