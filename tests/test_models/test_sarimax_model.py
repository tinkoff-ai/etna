import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from etna.models import SARIMAXModel
from etna.pipeline import Pipeline


def test_sarimax_forecaster_run(example_tsds):
    """
    Given: I have dataframe with 2 segments
    When:
    Then: I get 7 periods per dataset as a forecast
    """

    horizon = 7
    model = SARIMAXModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_sarimax_save_regressors_on_fit(example_reg_tsds):
    model = SARIMAXModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_sarimax_select_regressors_correctly(example_reg_tsds):
    model = SARIMAXModel()
    model.fit(ts=example_reg_tsds)
    for segment, segment_model in model._models.items():
        segment_features = example_reg_tsds[:, segment, :].droplevel("segment", axis=1)
        segment_regressors_expected = segment_features[example_reg_tsds.regressors]
        segment_regressors = segment_model._select_regressors(df=segment_features.reset_index())
        assert (segment_regressors == segment_regressors_expected).all().all()


def test_sarimax_forecaster_run_with_reg(example_reg_tsds):
    """
    Given: I have dataframe with 2 segments
    When:
    Then: I get 7 periods per dataset as a forecast
    """
    horizon = 7
    model = SARIMAXModel()
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_sarimax_forececaster_run_with_reg_custom_order(example_reg_tsds):
    """
    Given: I have dataframe with 2 segments
    When: Sarimax have non standard `order` param
    Then: I get 7 periods per dataset as a forecast
    """
    horizon = 7
    model = SARIMAXModel(order=(3, 1, 0))
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_prediction_interval_run_insample(example_tsds):
    model = SARIMAXModel()
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        # N.B. inplace forecast will not change target values, because `combine_first` in `SARIMAXModel.forecast` only fill nan values
        # assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        # assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_prediction_interval_run_infuture(example_tsds):
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


def test_forecast_raise_error_if_not_fitted(example_tsds):
    """Test that SARIMAX raise error when calling forecast without being fit."""
    model = SARIMAXModel()
    with pytest.raises(ValueError, match="model is not fitted!"):
        _ = model.forecast(ts=example_tsds)


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
        assert isinstance(models_dict[segment], SARIMAX)
