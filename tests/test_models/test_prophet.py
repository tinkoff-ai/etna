import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from prophet.serialize import model_to_dict

from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.models.prophet import _ProphetAdapter
from etna.pipeline import Pipeline
from tests.test_models.utils import assert_model_equals_loaded_original


def test_run(new_format_df):
    df = new_format_df

    ts = TSDataset(df, "1d")

    model = ProphetModel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


def test_run_with_reg(new_format_df, new_format_exog):
    df = new_format_df

    regressors = new_format_exog.copy()
    regressors.columns.set_levels(["regressor_exog"], level="feature", inplace=True)
    regressors_floor = new_format_exog.copy()
    regressors_floor.columns.set_levels(["floor"], level="feature", inplace=True)
    regressors_cap = regressors_floor.copy() + 1
    regressors_cap.columns.set_levels(["cap"], level="feature", inplace=True)
    exog = pd.concat([regressors, regressors_floor, regressors_cap], axis=1)

    ts = TSDataset(df, "1d", df_exog=exog, known_future="all")

    model = ProphetModel(growth="logistic")
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


def test_run_with_cap_floor():
    cap = 101
    floor = -1

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=100),
            "segment": "segment_0",
            "target": list(range(100)),
        }
    )
    df_exog = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2020-01-01", periods=120),
            "segment": "segment_0",
            "cap": cap,
            "floor": floor,
        }
    )
    ts = TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog), freq="D", known_future="all")

    model = ProphetModel(growth="logistic")
    pipeline = Pipeline(model=model, horizon=7)
    pipeline.fit(ts)

    ts_future = pipeline.forecast()
    df_future = ts_future.to_pandas(flatten=True)

    assert np.all(df_future["target"] < cap)


def test_prediction_interval_run_insample(example_tsds):
    model = ProphetModel()
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_prediction_interval_run_infuture(example_tsds):
    model = ProphetModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_prophet_save_regressors_on_fit(example_reg_tsds):
    model = ProphetModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_get_model_before_training():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = ProphetModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_after_training(example_tsds):
    """Check that get_model method returns dict of objects of Prophet class."""
    pipeline = Pipeline(model=ProphetModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], Prophet)


@pytest.fixture
def prophet_default_params():
    params = {
        "growth": "linear",
        "changepoints": None,
        "n_changepoints": 25,
        "changepoint_range": 0.8,
        "yearly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto",
        "holidays": None,
        "seasonality_mode": "additive",
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.05,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000,
        "stan_backend": None,
        "additional_seasonality_params": (),
    }
    return params


def test_getstate_not_fitted(prophet_default_params):
    model = _ProphetAdapter()
    state = model.__getstate__()
    expected_state = {
        "_is_fitted": False,
        "_model_dict": {},
        "regressor_columns": None,
        **prophet_default_params,
    }
    assert state == expected_state


def test_getstate_fitted(example_tsds, prophet_default_params):
    model = _ProphetAdapter()
    df = example_tsds.to_pandas()["segment_1"].reset_index()
    model.fit(df, regressors=[])
    state = model.__getstate__()
    expected_state = {
        "_is_fitted": True,
        "_model_dict": model_to_dict(model.model),
        "regressor_columns": [],
        **prophet_default_params,
    }
    assert state == expected_state


def test_setstate_not_fitted():
    model_1 = _ProphetAdapter(n_changepoints=25)
    initial_state = model_1.__getstate__()

    model_2 = _ProphetAdapter(n_changepoints=20)
    model_2.__setstate__(initial_state)
    new_state = model_2.__getstate__()
    assert new_state == initial_state


def test_setstate_fitted(example_tsds):
    model_1 = _ProphetAdapter()
    df = example_tsds.to_pandas()["segment_1"].reset_index()
    model_1.fit(df, regressors=[])
    initial_state = model_1.__getstate__()

    model_2 = _ProphetAdapter()
    model_2.__setstate__(initial_state)
    new_state = model_2.__getstate__()
    assert new_state == initial_state


@pytest.mark.xfail(reason="Non native serialization, should be fixed in inference-v2.0")
def test_save_load(example_tsds):
    model = ProphetModel()
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.mark.parametrize(
    "custom_seasonality",
    (
        [{"name": "s1", "period": 14, "fourier_order": 3}],
        [{"name": "s1", "period": 14, "fourier_order": 3}, {"name": "s2", "period": 10, "fourier_order": 3}],
    ),
)
def test_custom_seasonality(custom_seasonality):
    model = ProphetModel(additional_seasonality_params=custom_seasonality)
    for seasonality in custom_seasonality:
        assert seasonality["name"] in model._base_model.model.seasonalities
