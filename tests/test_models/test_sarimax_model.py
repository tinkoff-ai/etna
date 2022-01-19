import pytest

from etna.models import SARIMAXModel


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


def test_sarimax_forecaster_run_with_reg_case2(example_reg_tsds):
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
    with pytest.raises(ValueError, match="The model is not fitted yet"):
        _ = model.forecast(ts=example_tsds)
