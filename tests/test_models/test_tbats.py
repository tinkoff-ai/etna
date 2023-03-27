from copy import deepcopy

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


@pytest.fixture()
def small_periodic_ts(periodic_ts):
    df = periodic_ts[0].df.loc[:, pd.IndexSlice["segment_1", :]].iloc[-10:]
    ts = TSDataset(df, freq="D")
    return ts


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
@pytest.mark.parametrize("method", ("forecast", "predict"))
def test_not_fitted(model, method, linear_segments_ts_unique):
    train, test = linear_segments_ts_unique
    to_forecast = train.make_future(3)

    method_to_call = getattr(model, method)
    with pytest.raises(ValueError, match="model is not fitted!"):
        method_to_call(ts=to_forecast)


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_forecast_format(model, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")
    lags = LagTransform(lags=[3, 4, 5], in_column="target")
    ts.fit_transform([lags])
    model.fit(ts)
    future_ts = ts.make_future(3, transforms=[lags])
    model.forecast(future_ts)
    future_ts.inverse_transform([lags])
    assert not future_ts.isnull().values.any()


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_predict_format(model, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")
    lags = LagTransform(lags=[3], in_column="target")
    ts.fit_transform([lags])
    model.fit(ts)
    pred = model.predict(ts)
    assert not pred[:, :, "target"].isnull().values.any()


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
@pytest.mark.parametrize("method, use_future", (("predict", False), ("forecast", True)))
def test_dummy(model, method, use_future, sinusoid_ts):
    train, test = sinusoid_ts
    model.fit(train)

    if use_future:
        pred_ts = train.make_future(14)
        y_true = test
    else:
        pred_ts = deepcopy(train)
        y_true = train

    method_to_call = getattr(model, method)
    y_pred = method_to_call(ts=pred_ts)

    metric = MAE("macro")
    value_metric = metric(y_true, y_pred)
    assert value_metric < 0.33


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
@pytest.mark.parametrize("method, use_future", (("predict", False), ("forecast", True)))
def test_prediction_interval(model, method, use_future, example_tsds):
    model.fit(example_tsds)
    if use_future:
        pred_ts = example_tsds.make_future(3)
    else:
        pred_ts = deepcopy(example_tsds)

    method_to_call = getattr(model, method)
    forecast = method_to_call(ts=pred_ts, prediction_interval=True, quantiles=[0.025, 0.975])

    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


@pytest.mark.long_2
@pytest.mark.parametrize("model", [TBATSModel(), BATSModel()])
def test_save_load(model, example_tsds):
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.mark.parametrize("method", ("predict_components", "forecast_components"))
def test_decompose_not_fitted(small_periodic_ts, method):
    model = _TBATSAdapter(model=BATS())

    method_to_call = getattr(model, method)
    with pytest.raises(ValueError, match="Model is not fitted!"):
        method_to_call(df=small_periodic_ts.df)


@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
def test_decompose_forecast_output_format(small_periodic_ts, estimator):
    horizon = 3
    model = estimator()
    model.fit(small_periodic_ts)

    components = model._models["segment_1"]._decompose_forecast(horizon=horizon)
    assert isinstance(components, np.ndarray)
    assert components.shape[0] == horizon


@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
def test_decompose_predict_output_format(small_periodic_ts, estimator):
    model = estimator()
    model.fit(small_periodic_ts)

    target = small_periodic_ts[:, "segment_1", "target"].values
    components = model._models["segment_1"]._decompose_predict()
    assert isinstance(components, np.ndarray)
    assert components.shape[0] == target.shape[0]


@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
def test_named_components_output_format(small_periodic_ts, estimator):
    horizon = 3
    model = estimator()
    model.fit(small_periodic_ts)

    segment_model = model._models["segment_1"]
    components = segment_model._decompose_forecast(horizon=horizon)
    components = segment_model._process_components(raw_components=components)

    assert isinstance(components, pd.DataFrame)
    assert len(components) == horizon


@pytest.mark.parametrize(
    "train_slice,decompose_slice", ((slice(5, 20), slice(None, 20)), (slice(5, 10), slice(10, 20)))
)
def test_predict_components_out_of_sample_error(periodic_dfs, train_slice, decompose_slice):
    train, _ = periodic_dfs

    model = _TBATSAdapter(model=BATS())
    model.fit(train.iloc[train_slice], [])
    with pytest.raises(NotImplementedError, match="isn't currently implemented for out-of-sample prediction"):
        model.predict_components(df=train.iloc[decompose_slice])


@pytest.mark.long_1
@pytest.mark.parametrize(
    "estimator,params,components_names",
    (
        (
            BATSModel,
            {"use_box_cox": False, "use_trend": True, "use_arma_errors": True, "seasonal_periods": [7, 14]},
            {
                "target_component_local_level",
                "target_component_trend",
                "target_component_arma(p=1,q=1)",
                "target_component_seasonal(s=7)",
                "target_component_seasonal(s=14)",
            },
        ),
        (
            TBATSModel,
            {"use_box_cox": False, "use_trend": True, "use_arma_errors": False, "seasonal_periods": [7, 14]},
            {
                "target_component_local_level",
                "target_component_trend",
                "target_component_seasonal(s=7.0)",
                "target_component_seasonal(s=14.0)",
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "method,use_future",
    (
        ("predict_components", False),
        ("forecast_components", True),
    ),
)
def test_components_names(periodic_ts, estimator, params, components_names, method, use_future):
    train, test = periodic_ts
    model = estimator(**params)
    model.fit(train)

    pred_ts = train.make_future(3) if use_future else train

    for segment in test.columns.get_level_values("segment"):
        pred_df = pred_ts[:, segment, :].droplevel("segment", axis=1).reset_index()
        method_to_call = getattr(model._models[segment], method)
        components_df = method_to_call(df=pred_df)
        assert set(components_df.columns) == components_names


@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
@pytest.mark.parametrize("method,use_future", (("predict_components", False), ("forecast_components", True)))
def test_seasonal_components_not_fitted(small_periodic_ts, estimator, method, use_future):
    model = estimator(seasonal_periods=[7, 14], use_arma_errors=False)
    model.fit(small_periodic_ts)

    segment_model = model._models["segment_1"]
    segment_model._fitted_model.params.components.seasonal_periods = []

    pred_ts = small_periodic_ts
    if use_future:
        pred_ts = pred_ts.make_future(3)

    pred_df = pred_ts[:, "segment_1", :].droplevel("segment", axis=1).reset_index()

    method_to_call = getattr(segment_model, method)
    with pytest.warns(Warning, match=f"Following components are not fitted: Seasonal!"):
        method_to_call(df=pred_df)


@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
@pytest.mark.parametrize("method,use_future", (("predict_components", False), ("forecast_components", True)))
def test_arma_component_not_fitted(small_periodic_ts, estimator, method, use_future):
    model = estimator(use_arma_errors=True, seasonal_periods=[])
    model.fit(small_periodic_ts)

    segment_model = model._models["segment_1"]
    segment_model._fitted_model.params.components.use_arma_errors = False

    pred_ts = small_periodic_ts
    if use_future:
        pred_ts = pred_ts.make_future(3)

    pred_df = pred_ts[:, "segment_1", :].droplevel("segment", axis=1).reset_index()

    method_to_call = getattr(segment_model, method)
    with pytest.warns(Warning, match=f"Following components are not fitted: ARMA!"):
        method_to_call(df=pred_df)


@pytest.mark.parametrize(
    "estimator",
    (
        BATSModel,
        TBATSModel,
    ),
)
@pytest.mark.parametrize("method,use_future", (("predict_components", False), ("forecast_components", True)))
def test_arma_w_seasonal_components_not_fitted(small_periodic_ts, estimator, method, use_future):
    model = estimator(use_arma_errors=True, seasonal_periods=[2, 3])
    model.fit(small_periodic_ts)

    segment_model = model._models["segment_1"]
    segment_model._fitted_model.params.components.use_arma_errors = False
    segment_model._fitted_model.params.components.seasonal_periods = []

    pred_ts = small_periodic_ts
    if use_future:
        pred_ts = pred_ts.make_future(3)

    pred_df = pred_ts[:, "segment_1", :].droplevel("segment", axis=1).reset_index()

    method_to_call = getattr(segment_model, method)
    with pytest.warns(Warning, match=f"Following components are not fitted: Seasonal, ARMA!"):
        method_to_call(df=pred_df)


@pytest.mark.long_1
@pytest.mark.filterwarnings("ignore:.*not fitted.*")
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
@pytest.mark.parametrize("method,use_future", (("predict_components", False), ("forecast_components", True)))
def test_forecast_decompose_sum_up_to_target(periodic_ts, estimator, params, method, use_future):
    train, test = periodic_ts

    model = estimator(**params)
    model.fit(train)

    if use_future:
        pred_ts = train.make_future(future_steps=14)
        y_pred = model.forecast(pred_ts)

    else:
        pred_ts = deepcopy(train)
        y_pred = model.predict(train)

    for segment in y_pred.columns.get_level_values("segment"):
        pred_df = pred_ts[:, segment, :].droplevel("segment", axis=1).reset_index()

        method_to_call = getattr(model._models[segment], method)

        components = method_to_call(df=pred_df)

        y_hat_pred = np.sum(components.values, axis=1)
        np.testing.assert_allclose(y_hat_pred, y_pred[:, segment, "target"].values)
