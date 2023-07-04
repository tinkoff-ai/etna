from unittest.mock import Mock

import pandas as pd
import pytest
from lightning_fabric.utilities.seed import seed_everything

from etna.metrics import MAE
from etna.models.nn import TFTModel
from etna.models.nn.utils import PytorchForecastingDatasetBuilder
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


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
    ts_pred = ts_train.make_future(future_steps=horizon, transforms=[dft], tail_steps=encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts=ts_pred, prediction_size=horizon)

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
    ts_pred = ts_train.make_future(future_steps=horizon, transforms=[std, dft], tail_steps=encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts=ts_pred, prediction_size=horizon)
    ts_pred.inverse_transform([std, dft])
    mae = MAE("macro")
    assert mae(ts_test, ts_pred) < 0.24


def test_prediction_interval_run_infuture(example_tsds):
    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)
    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=8), lr=0.1)
    model.fit(example_tsds)
    future = example_tsds.make_future(future_steps=horizon, tail_steps=pfdb.max_encoder_length)
    forecast = model.forecast(ts=future, prediction_size=horizon, prediction_interval=True, quantiles=[0.02, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.02", "target_0.98", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.98"] - segment_slice["target_0.02"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.02"] >= 0).all()
        assert (segment_slice["target_0.98"] - segment_slice["target"] >= 0).all()


def test_prediction_interval_run_infuture_warning_not_found_quantiles(example_tsds):
    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)
    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=2), lr=0.1)
    model.fit(example_tsds)
    future = example_tsds.make_future(future_steps=horizon, tail_steps=pfdb.max_encoder_length)
    with pytest.warns(UserWarning, match="Quantiles: \[0.4\] can't be computed"):
        forecast = model.forecast(
            ts=future, prediction_size=horizon, prediction_interval=True, quantiles=[0.02, 0.4, 0.98]
        )
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
    future = example_tsds.make_future(future_steps=horizon, tail_steps=pfdb.max_encoder_length)
    with pytest.warns(UserWarning, match="Quantiles can't be computed"):
        forecast = model.forecast(ts=future, prediction_size=horizon, prediction_interval=True, quantiles=[0.02, 0.98])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target"}.issubset(segment_slice.columns)
        assert {"target_0.02", "target_0.98"}.isdisjoint(segment_slice.columns)


def test_forecast_model_equals_pipeline(example_tsds):
    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)

    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=8), lr=0.1)
    seed_everything(0)
    model.fit(example_tsds)
    future = example_tsds.make_future(future_steps=horizon, tail_steps=pfdb.max_encoder_length)
    forecast_model = model.forecast(
        ts=future, prediction_size=horizon, prediction_interval=True, quantiles=[0.02, 0.98]
    )

    model = TFTModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=8), lr=0.1)
    pipeline = Pipeline(model=model, transforms=[], horizon=horizon)
    seed_everything(0)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast(prediction_interval=True, quantiles=[0.02, 0.98])

    pd.testing.assert_frame_equal(forecast_model.to_pandas(), forecast_pipeline.to_pandas())


def test_save_load(example_tsds):
    horizon = 3
    pfdb = _get_default_dataset_builder(horizon)
    model = TFTModel(
        dataset_builder=pfdb,
        lr=0.1,
        trainer_params=dict(max_epochs=2),
        train_batch_size=64,
    )
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=horizon)


def test_repr():
    model = TFTModel(
        decoder_length=3, encoder_length=4, lr=0.1, trainer_params=dict(max_epochs=2, gpus=0), train_batch_size=64
    )
    assert (
        repr(model) == "TFTModel(decoder_length = 3, encoder_length = 4, "
        "dataset_builder = PytorchForecastingDatasetBuilder(max_encoder_length = 4, min_encoder_length = 4, "
        "min_prediction_idx = None, min_prediction_length = None, max_prediction_length = 3, static_categoricals = [], "
        "static_reals = [], time_varying_known_categoricals = [], time_varying_known_reals = ['time_idx'], "
        "time_varying_unknown_categoricals = [], time_varying_unknown_reals = ['target'], variable_groups = {}, "
        "constant_fill_strategy = [], allow_missing_timesteps = True, lags = {}, add_relative_time_idx = True, "
        "add_target_scales = True, add_encoder_length = True, target_normalizer = None, categorical_encoders = {}, scalers = {}, ), "
        "train_batch_size = 64, test_batch_size = 64, lr = 0.1, hidden_size = 16, lstm_layers = 1, "
        "attention_head_size = 4, dropout = 0.1, hidden_continuous_size = 8, "
        "loss = QuantileLoss(), trainer_params = {'max_epochs': 2, 'gpus': 0}, quantiles_kwargs = {}, )"
    )


def test_tft_forecast_throw_error_on_return_components():
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        TFTModel.forecast(self=Mock(), ts=Mock(), prediction_size=Mock(), return_components=True)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = TFTModel(decoder_length=3, encoder_length=4, trainer_params=dict(max_epochs=1, gpus=0))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
