from copy import deepcopy

import pytest
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoCES
from statsforecast.models import AutoETS
from statsforecast.models import AutoTheta

from etna.libs.statsforecast import ARIMA
from etna.models import StatsForecastARIMAModel
from etna.models import StatsForecastAutoARIMAModel
from etna.models import StatsForecastAutoCESModel
from etna.models import StatsForecastAutoETSModel
from etna.models import StatsForecastAutoThetaModel
from etna.pipeline import Pipeline
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


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


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_save_regressors_on_fit(model, example_reg_tsds):
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_select_regressors_correctly(model, example_reg_tsds):
    model.fit(ts=example_reg_tsds)
    for segment, segment_model in model._models.items():
        segment_features = example_reg_tsds[:, segment, :].droplevel("segment", axis=1)
        segment_regressors_expected = segment_features[example_reg_tsds.regressors]
        segment_regressors = segment_model._select_regressors(df=segment_features.reset_index())
        assert (segment_regressors == segment_regressors_expected).all().all()


@pytest.mark.parametrize("method_name", ["forecast", "predict"])
@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_prediction_raise_error_if_not_fitted(model, method_name, example_tsds):
    with pytest.raises(ValueError, match="model is not fitted!"):
        method = getattr(model, method_name)
        _ = method(ts=example_tsds)


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_get_model_before_training(model):
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = model.get_model()


@pytest.mark.parametrize(
    "model, expected_type",
    [
        (StatsForecastARIMAModel(), ARIMA),
        (StatsForecastAutoARIMAModel(), AutoARIMA),
        (StatsForecastAutoCESModel(), AutoCES),
        (StatsForecastAutoETSModel(), AutoETS),
        (StatsForecastAutoThetaModel(), AutoTheta),
    ],
)
def test_get_model_after_training(model, expected_type, example_tsds):
    pipeline = Pipeline(model=model)
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], expected_type)


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_predict_train(model, example_tsds, recwarn):
    model.fit(example_tsds)
    res = model.predict(example_tsds)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == len(example_tsds.index) * 2


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_predict_before_train_fail(model, example_tsds):
    train_ts = deepcopy(example_tsds)
    train_ts.df = train_ts.df.iloc[10:]
    before_train_ts = deepcopy(example_tsds)
    model.fit(train_ts)

    with pytest.raises(NotImplementedError, match="This model can't make predict on past out-of-sample data"):
        _ = model.predict(before_train_ts)


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_predict_future_fail(model, example_tsds):
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=7)

    with pytest.raises(NotImplementedError, match="This model can't make predict on future out-of-sample data"):
        _ = model.predict(future_ts)


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_forecast_future(model, example_tsds):
    horizon = 7
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == horizon * 2


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_forecast_future_with_gap_fail(model, example_tsds):
    horizon = 7
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    future_ts.df = future_ts.df.iloc[2:]

    with pytest.raises(
        NotImplementedError,
        match="This model can't make forecast on out-of-sample data that goes after training data with a gap",
    ):
        _ = model.forecast(future_ts)


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_forecast_train_fail(model, example_tsds):
    model.fit(example_tsds)
    with pytest.raises(NotImplementedError, match="This model can't make forecast on history data"):
        _ = model.forecast(example_tsds)


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_predict_with_interval(model, example_tsds):
    model.fit(example_tsds)
    forecast = model.predict(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_forecast_with_interval(model, example_tsds):
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(7)
    forecast = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


@pytest.mark.parametrize(
    "model",
    [
        StatsForecastARIMAModel(),
        StatsForecastAutoARIMAModel(),
        StatsForecastAutoCESModel(),
        StatsForecastAutoETSModel(),
        StatsForecastAutoThetaModel(),
    ],
)
def test_save_load(model, example_tsds):
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


@pytest.mark.parametrize(
    "model, expected_length",
    [
        (StatsForecastARIMAModel(), 3),
        (StatsForecastARIMAModel(season_length=7), 6),
        (StatsForecastAutoARIMAModel(), 0),
        (StatsForecastAutoCESModel(), 0),
        (StatsForecastAutoETSModel(), 0),
        (StatsForecastAutoThetaModel(), 0),
    ],
)
def test_params_to_tune(model, expected_length, example_tsds):
    ts = example_tsds
    assert len(model.params_to_tune()) == expected_length
    assert_sampling_is_valid(model=model, ts=ts)
