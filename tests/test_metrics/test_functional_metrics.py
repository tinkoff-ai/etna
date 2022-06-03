import pytest

from etna.metrics import mae
from etna.metrics import mape
from etna.metrics import medae
from etna.metrics import mse
from etna.metrics import msle
from etna.metrics import r2_score
from etna.metrics import sign
from etna.metrics import smape


@pytest.fixture()
def right_mae_value():
    pass


@pytest.fixture()
def y_true_1d():
    return [1, 1]


@pytest.fixture()
def y_pred_1d():
    return [2, 2]


@pytest.mark.parametrize(
    "metric, right_metrics_value",
    ((mae, 1), (mse, 1), (mape, 100), (smape, 66.6666666667), (medae, 1), (r2_score, 0), (sign, -1)),
)
def test_all_1d_metrics(metric, right_metrics_value, y_true_1d, y_pred_1d):
    assert round(metric(y_true_1d, y_pred_1d), 10) == right_metrics_value


def test_mle_metric_exception(y_true_1d, y_pred_1d):
    y_true_1d[-1] = -1

    with pytest.raises(ValueError):
        msle(y_true_1d, y_pred_1d)


@pytest.fixture()
def y_true_2d():
    return [[1, 1], [1, 1]]


@pytest.fixture()
def y_pred_2d():
    return [[2, 2], [2, 2]]


@pytest.mark.parametrize(
    "metric, right_metrics_value",
    ((mae, 1), (mse, 1), (mape, 100), (smape, 66.6666666667), (medae, 1), (r2_score, 0.0), (sign, -1)),
)
def test_all_2d_metrics(metric, right_metrics_value, y_true_2d, y_pred_2d):
    assert round(metric(y_true_2d, y_pred_2d), 10) == right_metrics_value
