import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models.tbats import BATS
from etna.models.tbats import BATSModel
from etna.models.tbats import TBATSModel
from etna.models.tbats import _TBATSAdapter
from etna.transforms import LagTransform
from tests.test_models.test_linear_model import linear_segments_by_parameters
from tests.test_models.utils import assert_model_equals_loaded_original


@pytest.fixture()
def linear_segments_ts_unique(random_seed):
    alpha_values = [np.random.rand() * 4 - 2 for _ in range(3)]
    intercept_values = [np.random.rand() * 4 + 1 for _ in range(3)]
    return linear_segments_by_parameters(alpha_values, intercept_values)


@pytest.fixture()
def sinusoid_ts():
    horizon = 14
    periods = 100
    sinusoid_ts_1 = pd.DataFrame(
        {
            "segment": np.zeros(periods),
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": [np.sin(i) for i in range(periods)],
        }
    )
    sinusoid_ts_2 = pd.DataFrame(
        {
            "segment": np.ones(periods),
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": [np.sin(i + 1) for i in range(periods)],
        }
    )
    df = pd.concat((sinusoid_ts_1, sinusoid_ts_2))
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts.train_test_split(test_size=horizon)


@pytest.fixture()
def periodic_ts():
    horizon = 14
    periods = 100
    t = np.arange(periods)

    # data from https://pypi.org/project/tbats/
    y = (
        5 * np.sin(t * 2 * np.pi / 7)
        + 2 * np.cos(t * 2 * np.pi / 14)
        + ((t / 20) ** 1.5 + np.random.normal(size=periods) * t / 50)
        + 20
    )

    ts_1 = pd.DataFrame(
        {
            "segment": ["segment_1"] * periods,
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": y,
        }
    )
    ts_2 = pd.DataFrame(
        {
            "segment": ["segment_2"] * periods,
            "timestamp": pd.date_range(start="1/1/2018", periods=periods),
            "target": 2 * y,
        }
    )
    df = pd.concat((ts_1, ts_2))
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts.train_test_split(test_size=horizon)


@pytest.mark.parametrize(
    "model_class, model_class_repr",
    ((TBATSModel, "TBATSModel"), (BATSModel, "BATSModel")),
)
def test_repr(model_class, model_class_repr):
    kwargs = {
        "use_box_cox": None,
        "box_cox_bounds": None,
        "use_trend": None,
        "use_damped_trend": None,
        "seasonal_periods": None,
        "use_arma_errors": None,
        "show_warnings": None,
        "n_jobs": None,
        "multiprocessing_start_method": None,
        "context": None,
    }
    kwargs_repr = (
        "use_box_cox = None, "
        + "box_cox_bounds = None, "
        + "use_trend = None, "
        + "use_damped_trend = None, "
        + "seasonal_periods = None, "
        + "use_arma_errors = None, "
        + "show_warnings = None, "
        + "n_jobs = None, "
        + "multiprocessing_start_method = None, "
        + "context = None"
    )
    model = model_class(**kwargs)
    model_repr = model.__repr__()
    true_repr = f"{model_class_repr}({kwargs_repr}, )"
    assert model_repr == true_repr


@pytest.mark.parametrize("model", (TBATSModel(), BATSModel()))
def test_not_fitted(model, linear_segments_ts_unique):
    train, test = linear_segments_ts_unique
    to_forecast = train.make_future(3)
    with pytest.raises(ValueError, match="model is not fitted!"):
        model.forecast(to_forecast)


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_format(model, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")
    lags = LagTransform(lags=[3, 4, 5], in_column="target")
    ts.fit_transform([lags])
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    assert not future_ts.isnull().values.any()


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_dummy(model, sinusoid_ts):
    train, test = sinusoid_ts
    model.fit(train)
    future_ts = train.make_future(14)
    y_pred = model.forecast(future_ts)
    metric = MAE("macro")
    value_metric = metric(y_pred, test)
    assert value_metric < 0.33


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_prediction_interval(model, example_tsds):
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(3)
    forecast = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_save_load(model, example_tsds):
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_forecast_decompose_not_fitted():
    model = _TBATSAdapter(model=BATS())
    with pytest.raises(ValueError, match="Model is not fitted!"):
        model.forecast_components(horizon=5)


@pytest.mark.long_2
@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
def test_predict_components_not_implemented(periodic_ts, estimator):
    train, test = periodic_ts
    model = estimator()
    model.fit(train)

    with pytest.raises(NotImplementedError, match="Prediction decomposition isn't currently implemented!"):
        for segment in test.columns.get_level_values("segment"):
            model._models[segment].predict_components(horizon=3)


@pytest.mark.long_2
@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
def test_decompose_forecast_output_format(periodic_ts, estimator):
    horizon = 5
    train, test = periodic_ts
    model = estimator()
    model.fit(train)

    for segment in test.columns.get_level_values("segment"):
        components = model._models[segment]._decompose_forecast(horizon=horizon)
        assert isinstance(components, np.ndarray)
        assert components.shape[0] == horizon


@pytest.mark.long_2
@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
def test_named_components_output_format(periodic_ts, estimator):
    horizon = 5
    train, test = periodic_ts
    model = estimator()
    model.fit(train)

    for segment in test.columns.get_level_values("segment"):
        components = model._models[segment]._decompose_forecast(horizon=horizon)
        named_components = model._models[segment]._named_components(raw_components=components)

        for component in named_components.values():
            assert len(component) == horizon


@pytest.mark.long_2
@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
@pytest.mark.parametrize(
    "data,params,message",
    (
        (
            "sinusoid_ts",
            {"use_box_cox": False, "use_trend": False, "use_arma_errors": False, "seasonal_periods": [7, 14]},
            "Seasonal components is not fitted!",
        ),
        (
            "periodic_ts",
            {
                "use_box_cox": True,
                "use_trend": True,
                "use_damped_trend": True,
                "use_arma_errors": True,
                "seasonal_periods": [7, 14],
            },
            "ARMA components is not fitted!",
        ),
    ),
)
def test_fitted_components_warnings(data, estimator, params, message, request):
    train, test = request.getfixturevalue(data)

    model = estimator(**params)
    model.fit(train)

    with pytest.warns(Warning, match=message):
        for segment in test.columns.get_level_values("segment"):
            model._models[segment].forecast_components(horizon=3)


@pytest.mark.long_2
@pytest.mark.filterwarnings("ignore:.* is not fitted!$")
@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
@pytest.mark.parametrize(
    "params",
    (
        {"use_box_cox": False, "use_trend": False, "use_arma_errors": False},
        {"use_box_cox": False, "use_trend": False, "use_arma_errors": False, "seasonal_periods": [7, 14]},
        {"use_box_cox": False, "use_trend": True, "use_arma_errors": False},
        {"use_box_cox": False, "use_trend": False, "use_arma_errors": True},
        {"use_box_cox": False, "use_trend": True, "use_arma_errors": True, "seasonal_periods": [7, 14]},
        {
            "use_box_cox": False,
            "use_trend": True,
            "use_damped_trend": True,
            "use_arma_errors": True,
            "seasonal_periods": [7, 14],
        },
        {
            "use_box_cox": True,
            "use_trend": True,
            "use_damped_trend": True,
            "use_arma_errors": True,
            "seasonal_periods": [7, 14],
        },
    ),
)
def test_forecast_decompose(periodic_ts, estimator, params):
    train, test = periodic_ts

    horizon = 14
    model = estimator(**params)
    model.fit(train)
    future_ts = train.make_future(horizon)
    y_pred = model.forecast(future_ts)

    for segment in y_pred.columns.get_level_values("segment"):
        components = model._models[segment].forecast_components(horizon=horizon)
        y_hat_pred = np.sum(components.values, axis=1)
        np.testing.assert_allclose(y_hat_pred, y_pred[:, segment, "target"].values)
