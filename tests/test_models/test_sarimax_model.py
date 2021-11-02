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


def test_confidence_interval_run_insample(example_tsds):
    model = SARIMAXModel()
    model.fit(example_tsds)
    forecast = model.forecast(example_tsds, confidence_interval=True, interval_width=0.95)
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_lower", "target_upper", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_upper"] - segment_slice["target_lower"] >= 0).all()


def test_confidence_interval_run_infuture(example_tsds):
    model = SARIMAXModel()
    model.fit(example_tsds)
    future = example_tsds.make_future(10)
    forecast = model.forecast(future, confidence_interval=True, interval_width=0.95)
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_lower", "target_upper", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_upper"] - segment_slice["target_lower"] >= 0).all()
