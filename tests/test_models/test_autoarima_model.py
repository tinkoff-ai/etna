import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from etna.models import AutoARIMAModel
from etna.pipeline import Pipeline


def test_autoarima_forecaster_run(example_tsds):
    """
    Given: I have dataframe with 2 segments
    When:
    Then: I get 7 periods per dataset as a forecast
    """

    horizon = 7
    model = AutoARIMAModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_autoarima_save_regressors_on_fit(example_reg_tsds):
    model = AutoARIMAModel()
    model.fit(ts=example_reg_tsds)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == example_reg_tsds.regressors


def test_autoarima_select_regressors_correctly(example_reg_tsds):
    model = AutoARIMAModel()
    model.fit(ts=example_reg_tsds)
    for segment, segment_model in model._models.items():
        segment_features = example_reg_tsds[:, segment, :].to_pandas().droplevel("segment", axis=1)
        segment_regressors_expected = segment_features[example_reg_tsds.regressors]
        segment_regressors = segment_model._select_regressors(df=segment_features.reset_index())
        assert (segment_regressors == segment_regressors_expected).all().all()


def test_autoarima_forecaster_run_with_reg(example_reg_tsds):
    """
    Given: I have dataframe with 2 segments
    When:
    Then: I get 7 periods per dataset as a forecast
    """
    horizon = 7
    model = AutoARIMAModel()
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_autoarima_forececaster_run_with_params(example_reg_tsds):
    """
    Given: I have dataframe with 2 segments
    When: AutoARIMAModel have non default params
    Then: I get 7 periods per dataset as a forecast
    """
    horizon = 7
    model = AutoARIMAModel(
        start_p=3,
        start_q=3,
        max_p=4,
        max_d=4,
        max_q=5,
        start_P=2,
        start_Q=2,
        max_P=3,
        max_D=3,
        max_Q=2,
        max_order=6,
        m=2,
        seasonal=True,
    )
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_prediction_interval_run_insample_autoarima(example_tsds):
    model = AutoARIMAModel()
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :].to_pandas()[segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        # N.B. inplace forecast will not change target values, because `combine_first` in `SARIMAXModel.forecast` only fill nan values
        # assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        # assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_prediction_interval_run_infuture_autoarima(example_tsds):
    model = AutoARIMAModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    forecast = model.forecast(future, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :].to_pandas()[segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()


def test_forecast_raise_error_if_not_fitted_autoarima(example_tsds):
    """Test that AutoARIMA raise error when calling forecast without being fit."""
    model = AutoARIMAModel()
    with pytest.raises(ValueError, match="model is not fitted!"):
        _ = model.forecast(ts=example_tsds)


def test_get_model_before_training_autoarima():
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = AutoARIMAModel()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


def test_get_model_after_training(example_tsds):
    """Check that get_model method returns dict of objects of AutoARIMA class."""
    pipeline = Pipeline(model=AutoARIMAModel())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], SARIMAXResultsWrapper)


def test_autoarima_forecast_1_point(example_tsds):
    """Check that AutoARIMA work with 1 point forecast."""
    horizon = 1
    model = AutoARIMAModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    pred = model.forecast(future_ts)
    assert len(pred.df) == horizon
    pred_quantiles = model.forecast(future_ts, prediction_interval=True, quantiles=[0.025, 0.8])
    assert len(pred_quantiles.df) == horizon
