from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import median_absolute_error as medae
from sklearn.metrics import r2_score

from etna.metrics.base import Metric
from etna.metrics.base import MetricAggregationMode
from etna.metrics.functional_metrics import mape
from etna.metrics.functional_metrics import max_deviation
from etna.metrics.functional_metrics import rmse
from etna.metrics.functional_metrics import sign
from etna.metrics.functional_metrics import smape
from etna.metrics.functional_metrics import wape
from etna.metrics.intervals_metrics import Coverage
from etna.metrics.intervals_metrics import Width
from etna.metrics.metrics import MAE
from etna.metrics.metrics import MAPE
from etna.metrics.metrics import MSE
from etna.metrics.metrics import MSLE
from etna.metrics.metrics import R2
from etna.metrics.metrics import RMSE
from etna.metrics.metrics import SMAPE
from etna.metrics.metrics import WAPE
from etna.metrics.metrics import MaxDeviation
from etna.metrics.metrics import MedAE
from etna.metrics.metrics import Sign
from etna.metrics.utils import compute_metrics
