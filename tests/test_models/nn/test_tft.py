import pytest

from etna.metrics import MAE
from etna.models.nn import TFTModel
from etna.models.nn.utils import PytorchForecastingDatasetBuilder
from etna.transforms import DateFlagsTransform
from etna.transforms import StandardScalerTransform


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8, 21])
def test_tft_model_run_weekly_overfit(ts_dataset_weekly_function_with_horizon, horizon, encoder_length=21):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When:
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="dateflag")
    pfdb = PytorchForecastingDatasetBuilder(
        max_encoder_length=encoder_length,
        min_encoder_length=encoder_length,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["dateflag_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts_train.fit_transform([dft])

    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=300, gradient_clip_val=0.1), lr=0.1)
    ts_pred = ts_train.make_future(horizon, transforms=[dft], tail_steps=encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred, horizon=horizon)

    mae = MAE("macro")
    assert mae(ts_test, ts_pred) < 0.24


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8])
def test_tft_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon, encoder_length=21):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When: I use scale transformations
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    std = StandardScalerTransform(in_column="target")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflag")
    pfdb = PytorchForecastingDatasetBuilder(
        max_encoder_length=encoder_length,
        min_encoder_length=encoder_length,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflag_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )

    ts_train.fit_transform([std, dft])

    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=300, gradient_clip_val=0.1), lr=0.1)
    ts_pred = ts_train.make_future(horizon, [std, dft], encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred, horizon=horizon)
    ts_pred.inverse_transform([std, dft])
    mae = MAE("macro")
    assert mae(ts_test, ts_pred) < 0.24


def _get_default_dataset_builder(horizon: int):
    return PytorchForecastingDatasetBuilder(
        max_encoder_length=21,
        min_encoder_length=21,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
        target_normalizer=None,
    )


def test_prediction_interval_run_infuture(example_tsds):
    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)
    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=8), lr=0.1)
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon, tail_steps=pfdb.max_encoder_length)
    forecast = model.forecast(future, horizon=horizon, prediction_interval=True, quantiles=[0.02, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.02", "target_0.98", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.98"] - segment_slice["target_0.02"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.02"] >= 0).all()
        assert (segment_slice["target_0.98"] - segment_slice["target"] >= 0).all()


def test_prediction_interval_run_infuture_warning_not_found_quantiles(example_tsds):
    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)
    example_tsds.fit_transform([])
    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=2), lr=0.1)
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon, [], pfdb.max_encoder_length)
    with pytest.warns(UserWarning, match="Quantiles: \[0.4\] can't be computed"):
        forecast = model.forecast(future, horizon=horizon, prediction_interval=True, quantiles=[0.02, 0.4, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.02", "target_0.98", "target"}.issubset(segment_slice.columns)
        assert {"target_0.4"}.isdisjoint(segment_slice.columns)


def test_prediction_interval_run_infuture_warning_loss(example_tsds):
    from pytorch_forecasting.metrics import MAE as MAEPF

    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)
    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=2), lr=0.1, loss=MAEPF())
    model.fit(example_tsds)
    future = example_tsds.make_future(horizon, tail_steps=pfdb.max_encoder_length)
    with pytest.warns(UserWarning, match="Quantiles can't be computed"):
        forecast = model.forecast(future, horizon, prediction_interval=True, quantiles=[0.02, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target"}.issubset(segment_slice.columns)
        assert {"target_0.02", "target_0.98"}.isdisjoint(segment_slice.columns)
