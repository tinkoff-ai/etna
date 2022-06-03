import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MSE
from etna.models.tbats import BATSPerSegmentModel
from etna.models.tbats import TBATSPerSegmentModel
from etna.transforms import LagTransform
from tests.test_models.test_linear_model import linear_segments_by_parameters


@pytest.fixture()
def linear_segments_ts_unique(random_seed):
    alpha_values = [np.random.rand() * 4 - 2 for _ in range(3)]
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)


@pytest.fixture()
def sinusoid_ts():
    horizon = 2
    sinusoid_ts_1 = pd.DataFrame(
        {
            "segment": np.zeros(8),
            "timestamp": pd.date_range(start="1/1/2018", end="1/08/2018"),
            "target": [np.sin(i) for i in range(8)],
        }
    )
    sinusoid_ts_2 = pd.DataFrame(
        {
            "segment": np.ones(8),
            "timestamp": pd.date_range(start="1/1/2018", end="1/08/2018"),
            "target": [np.sin(i + 1) for i in range(8)],
        }
    )
    df = pd.concat((sinusoid_ts_1, sinusoid_ts_2))
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts.train_test_split(test_size=horizon)


@pytest.mark.parametrize(
    "model_class, model_class_repr",
    ((TBATSPerSegmentModel, "TBATSPerSegmentModel"), (BATSPerSegmentModel, "BATSPerSegmentModel")),
)
def test_reper(model_class, model_class_repr):
    kwargs = {"seasonal_periods": [14, 30.5]}
    kwargs_repr = "seasonal_periods = [14, 30.5]"
    model = model_class(**kwargs)
    model_repr = model.__repr__()
    true_repr = f"{model_class_repr}({kwargs_repr}, )"
    assert model_repr == true_repr


@pytest.mark.parametrize("model", (TBATSPerSegmentModel(), BATSPerSegmentModel()))
def test_not_fitted(model, linear_segments_ts_unique):
    train, test = linear_segments_ts_unique
    to_forecast = train.make_future(3)
    with pytest.raises(ValueError, match="model is not fitted!"):
        model.forecast(to_forecast)


@pytest.mark.parametrize("model", [TBATSPerSegmentModel(), BATSPerSegmentModel()])
def test_format(model, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")
    lags = LagTransform(lags=[3, 4, 5], in_column="target")
    ts.fit_transform([lags])
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


@pytest.mark.parametrize("model", [TBATSPerSegmentModel(), BATSPerSegmentModel()])
def test_dummy(model, sinusoid_ts):
    train, test = sinusoid_ts
    model.fit(train)
    future_ts = train.make_future(2)
    y_pred = model.forecast(future_ts)
    metric = MSE(mode="macro")
    value_metric = metric(y_pred, test)
    assert value_metric < 4


@pytest.mark.parametrize("model", [TBATSPerSegmentModel(), BATSPerSegmentModel()])
def test_prediction_interval(model, example_tsds):
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()
