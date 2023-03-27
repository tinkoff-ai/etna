from copy import deepcopy

import numpy as np
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from etna.models import SARIMAXModel
from etna.models.sarimax import _SARIMAXAdapter
from etna.pipeline import Pipeline
from tests.test_models.utils import assert_model_equals_loaded_original


def _check_forecast(ts, model, horizon):
    model.fit(ts)
    future_ts = ts.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == horizon * 2


def _check_predict(ts, model):
    model.fit(ts)
    res = model.predict(ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == len(ts.index) * 2


def test_prediction(example_tsds):
    _check_forecast(ts=deepcopy(example_tsds), model=SARIMAXModel(), horizon=7)
    _check_predict(ts=deepcopy(example_tsds), model=SARIMAXModel())


def test_save_regressors_on_fit(example_reg_tsds):
    model = SARIMAXModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_select_regressors_correctly(example_reg_tsds):
    model = SARIMAXModel()
    model.fit(ts=example_reg_tsds)
    for segment, segment_model in model._models.items():
        segment_features = example_reg_tsds[:, segment, :].droplevel("segment", axis=1)
        segment_regressors_expected = segment_features[example_reg_tsds.regressors]
        segment_regressors = segment_model._select_regressors(df=segment_features.reset_index())
        assert (segment_regressors == segment_regressors_expected).all().all()


def test_prediction_with_simple_differencing(example_tsds):
    _check_forecast(ts=deepcopy(example_tsds), model=SARIMAXModel(simple_differencing=True), horizon=7)
    _check_predict(ts=deepcopy(example_tsds), model=SARIMAXModel(simple_differencing=True))


def test_prediction_with_reg(example_reg_tsds):
    _check_forecast(ts=deepcopy(example_reg_tsds), model=SARIMAXModel(), horizon=7)
    _check_predict(ts=deepcopy(example_reg_tsds), model=SARIMAXModel())


def test_prediction_with_reg_custom_order(example_reg_tsds):
    _check_forecast(ts=deepcopy(example_reg_tsds), model=SARIMAXModel(order=(3, 1, 0)), horizon=7)
    _check_predict(ts=deepcopy(example_reg_tsds), model=SARIMAXModel(order=(3, 1, 0)))


@pytest.mark.parametrize("method_name", ["forecast", "predict"])
def test_prediction_interval_insample(example_tsds, method_name):
    model = SARIMAXModel()
    model.fit(example_tsds)
    method = getattr(model, method_name)
    forecast = method(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        # N.B. inplace forecast will not change target values, because `combine_first` in `SARIMAXModel.forecast` only fill nan values
        # assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        # assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_forecast_prediction_interval_infuture(example_tsds):
    model = SARIMAXModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


@pytest.mark.parametrize("method_name", ["forecast", "predict"])
def test_prediction_raise_error_if_not_fitted(example_tsds, method_name):
    """Test that SARIMAX raise error when calling prediction without being fit."""
    model = SARIMAXModel()
    with pytest.raises(ValueError, match="model is not fitted!"):
        method = getattr(model, method_name)
        _ = method(ts=example_tsds)


def test_get_model_before_training():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = SARIMAXModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_after_training(example_tsds):
    """Check that get_model method returns dict of objects of SARIMAX class."""
    pipeline = Pipeline(model=SARIMAXModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], SARIMAXResultsWrapper)


def test_forecast_1_point(example_tsds):
    """Check that SARIMAX work with 1 point forecast."""
    horizon = 1
    model = SARIMAXModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    pred = model.forecast(future_ts)
    assert len(pred.df) == horizon
    pred_quantiles = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.8])
    assert len(pred_quantiles.df) == horizon


def test_save_load(example_tsds):
    model = SARIMAXModel()
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.mark.parametrize(
    "components_method_name,in_sample", (("predict_components", True), ("forecast_components", False))
)
def test_decomposition_hamiltonian_repr_error(dfs_w_exog, components_method_name, in_sample):
    train, test = dfs_w_exog
    pred_df = train if in_sample else test

    model = _SARIMAXAdapter(order=(2, 0, 0), seasonal_order=(1, 0, 0, 3), hamilton_representation=True)
    model.fit(train, ["f1", "f2"])

    components_method = getattr(model, components_method_name)

    with pytest.raises(
        ValueError, match="Prediction decomposition is not implemented for Hamilton representation of an ARMA!"
    ):
        _ = components_method(df=pred_df)


@pytest.mark.parametrize(
    "components_method_name,in_sample", (("predict_components", True), ("forecast_components", False))
)
@pytest.mark.parametrize(
    "regressors, regressors_components",
    (
        (["f1", "f2"], ["target_component_f1", "target_component_f2"]),
        (["f1"], ["target_component_f1"]),
        (["f1", "f1"], ["target_component_f1", "target_component_f1"]),
        ([], []),
    ),
)
@pytest.mark.parametrize("trend", (None, "t"))
def test_components_names(dfs_w_exog, regressors, regressors_components, trend, components_method_name, in_sample):
    expected_components = regressors_components + ["target_component_sarima"]

    train, test = dfs_w_exog
    pred_df = train if in_sample else test

    model = _SARIMAXAdapter(trend=trend)
    model.fit(train, regressors)

    components_method = getattr(model, components_method_name)
    components = components_method(df=pred_df)

    assert sorted(components.columns) == sorted(expected_components)


@pytest.mark.long_2
@pytest.mark.parametrize(
    "components_method_name,in_sample", (("predict_components", True), ("forecast_components", False))
)
@pytest.mark.parametrize(
    "mle_regression,time_varying_regression,regressors",
    (
        (True, False, ["f1", "f1"]),
        (True, False, []),
        (False, True, ["f1", "f2"]),
        (False, False, ["f1", "f2"]),
        (False, False, []),
    ),
)
@pytest.mark.parametrize("trend", (None, "t"))
@pytest.mark.parametrize("enforce_stationarity", (True, False))
@pytest.mark.parametrize("enforce_invertibility", (True, False))
@pytest.mark.parametrize("concentrate_scale", (True, False))
@pytest.mark.parametrize("use_exact_diffuse", (True, False))
def test_components_sum_up_to_target(
    dfs_w_exog,
    components_method_name,
    in_sample,
    mle_regression,
    time_varying_regression,
    trend,
    enforce_stationarity,
    enforce_invertibility,
    concentrate_scale,
    use_exact_diffuse,
    regressors,
):
    train, test = dfs_w_exog

    model = _SARIMAXAdapter(
        trend=trend,
        mle_regression=mle_regression,
        time_varying_regression=time_varying_regression,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        concentrate_scale=concentrate_scale,
        use_exact_diffuse=use_exact_diffuse,
    )
    model.fit(train, regressors)

    components_method = getattr(model, components_method_name)

    pred_df = train if in_sample else test

    pred = model.predict(pred_df, prediction_interval=False, quantiles=[])
    components = components_method(df=pred_df)

    np.testing.assert_allclose(np.sum(components.values, axis=1), np.squeeze(pred))
