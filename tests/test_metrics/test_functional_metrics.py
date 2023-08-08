import numpy.testing as npt
import pytest

from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import max_deviation
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import rmse
from etna.metrics import sign
from etna.metrics import smape
from etna.metrics import wape


@pytest.fixture()
def right_mae_value():
    pass


@pytest.fixture()
def y_true_1d():
    return [1, 3]


@pytest.fixture()
def y_pred_1d():
    return [2, 4]


@pytest.mark.parametrize(
    "metric, right_metrics_value",
    (
        (mae, 1),
        (mse, 1),
        (rmse, 1),
        (mape, 66 + 2 / 3),
        (smape, 47.6190476),
        (medae, 1),
        (r2_score, 0),
        (sign, -1),
        (max_deviation, 2),
        (wape, 1 / 2),
    ),
)
def test_all_1d_metrics(metric, right_metrics_value, y_true_1d, y_pred_1d):
    npt.assert_almost_equal(metric(y_true_1d, y_pred_1d), right_metrics_value)


def test_mle_metric_exception(y_true_1d, y_pred_1d):
    y_true_1d[-1] = -1

    with pytest.raises(ValueError):
        msle(y_true_1d, y_pred_1d)


@pytest.mark.parametrize(
    "metric",
    (
        mape,
        smape,
        sign,
        max_deviation,
        wape,
    ),
)
def test_all_wrong_mode(metric, y_true_1d, y_pred_1d):
    with pytest.raises(NotImplementedError):
        metric(y_true_1d, y_pred_1d, multioutput="unknown")


@pytest.fixture()
def y_true_2d():
    return [[10, 1], [11, 2]]


@pytest.fixture()
def y_pred_2d():
    return [[11, 2], [10, 1]]


@pytest.mark.parametrize(
    "metric, right_metrics_value",
    (
        (mae, 1),
        (mse, 1),
        (rmse, 1),
        (mape, 42 + 3 / 11),
        (smape, 38.0952380),
        (medae, 1),
        (r2_score, -3),
        (sign, 0),
        (max_deviation, 2),
        (wape, 1 / 6),
    ),
)
def test_all_2d_metrics_joint(metric, right_metrics_value, y_true_2d, y_pred_2d):
    npt.assert_almost_equal(metric(y_true_2d, y_pred_2d), right_metrics_value)


@pytest.mark.parametrize(
    "metric, params, right_metrics_value",
    (
        (mae, {"multioutput": "raw_values"}, [1, 1]),
        (mse, {"multioutput": "raw_values"}, [1, 1]),
        (rmse, {"multioutput": "raw_values"}, [1, 1]),
        (mape, {"multioutput": "raw_values"}, [9.5454545, 75]),
        (smape, {"multioutput": "raw_values"}, [9.5238095, 66 + 2 / 3]),
        (medae, {"multioutput": "raw_values"}, [1, 1]),
        (r2_score, {"multioutput": "raw_values"}, [-3, -3]),
        (sign, {"multioutput": "raw_values"}, [0, 0]),
        (max_deviation, {"multioutput": "raw_values"}, [1, 1]),
        (wape, {"multioutput": "raw_values"}, [0.0952381, 2 / 3]),
    ),
)
def test_all_2d_metrics_per_output(metric, params, right_metrics_value, y_true_2d, y_pred_2d):
    npt.assert_almost_equal(metric(y_true_2d, y_pred_2d, **params), right_metrics_value)
