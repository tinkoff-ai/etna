from unittest.mock import Mock

import pandas as pd
import pytest
from lightning_fabric.utilities.seed import seed_everything
from pytorch_forecasting.data import GroupNormalizer

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import DeepARModel
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
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8, 21])
def test_deepar_model_run_weekly_overfit(weekly_period_df, horizon, encoder_length=21):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When:
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_start = sorted(set(weekly_period_df.timestamp))[-horizon]
    train, test = (
        weekly_period_df[lambda x: x.timestamp < ts_start],
        weekly_period_df[lambda x: x.timestamp >= ts_start],
    )

    ts_train = TSDataset(TSDataset.to_dataset(train), "D")
    ts_test = TSDataset(TSDataset.to_dataset(test), "D")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflags")
    pfdb = PytorchForecastingDatasetBuilder(
        max_encoder_length=encoder_length,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflags_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    ts_train.fit_transform([dft])

    model = DeepARModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=300, gradient_clip_val=0.1), lr=0.1)
    ts_pred = ts_train.make_future(future_steps=horizon, transforms=[dft], tail_steps=encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts=ts_pred, prediction_size=horizon)

    mae = MAE("macro")

    assert mae(ts_test, ts_pred) < 0.2207


@pytest.mark.long_2
@pytest.mark.parametrize("horizon", [8])
def test_deepar_model_run_weekly_overfit_with_scaler(
    ts_dataset_weekly_function_with_horizon, horizon, encoder_length=21
):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When: I use scale transformations
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    std = StandardScalerTransform(in_column="target")
    dft = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="regressor_dateflags")
    pfdb = PytorchForecastingDatasetBuilder(
        max_encoder_length=encoder_length,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=["regressor_dateflags_day_number_in_week"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )

    ts_train.fit_transform([std, dft])

    model = DeepARModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=300, gradient_clip_val=0.1), lr=0.1)
    ts_pred = ts_train.make_future(future_steps=horizon, transforms=[std, dft], tail_steps=encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts=ts_pred, prediction_size=horizon)
    ts_pred.inverse_transform([std, dft])

    mae = MAE("macro")

    assert mae(ts_test, ts_pred) < 0.2207


@pytest.mark.parametrize("freq", ["1M", "1D", "A-DEC", "1B", "1H"])
def test_forecast_with_different_freq(weekly_period_df, freq):
    df = TSDataset.to_dataset(weekly_period_df)
    df.index = pd.Index(pd.date_range("2021-01-01", freq=freq, periods=len(df)), name="timestamp")

    ts = TSDataset(df, freq=freq)
    horizon = 20

    model_deepar = DeepARModel(
        encoder_length=horizon, decoder_length=horizon, trainer_params=dict(max_epochs=2), lr=0.01
    )
    pipeline_deepar = Pipeline(model=model_deepar, horizon=horizon)
    pipeline_deepar.fit(ts=ts)
    forecast = pipeline_deepar.forecast()

    assert len(forecast.df) == horizon
    assert pd.infer_freq(forecast.df.index) in {freq, freq[1:]}


def test_prediction_interval_run_infuture(example_tsds):
    horizon = 10
    model = DeepARModel(encoder_length=horizon, decoder_length=horizon, trainer_params=dict(max_epochs=2), lr=0.01)
    model.fit(example_tsds)
    future = example_tsds.make_future(future_steps=horizon, tail_steps=horizon)
    forecast = model.forecast(ts=future, prediction_size=horizon, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()


def test_forecast_model_equals_pipeline(example_tsds):
    horizon = 10
    pfdb = _get_default_dataset_builder(horizon)

    model = DeepARModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=2), lr=0.1)
    seed_everything(0)
    model.fit(example_tsds)
    future = example_tsds.make_future(future_steps=horizon, tail_steps=pfdb.max_encoder_length)
    forecast_model = model.forecast(
        ts=future, prediction_size=horizon, prediction_interval=True, quantiles=[0.02, 0.98]
    )

    model = DeepARModel(dataset_builder=pfdb, trainer_params=dict(max_epochs=2), lr=0.1)
    pipeline = Pipeline(model=model, transforms=[], horizon=horizon)
    seed_everything(0)
    pipeline.fit(example_tsds)
    forecast_pipeline = pipeline.forecast(prediction_interval=True, quantiles=[0.02, 0.98])

    pd.testing.assert_frame_equal(forecast_model.to_pandas(), forecast_pipeline.to_pandas())


def test_save_load(example_tsds):
    horizon = 3
    pfdb = _get_default_dataset_builder(horizon)
    model = DeepARModel(
        dataset_builder=pfdb,
        lr=0.1,
        trainer_params=dict(max_epochs=2),
        train_batch_size=64,
    )
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=horizon)


def test_repr():
    model = DeepARModel(
        decoder_length=3, encoder_length=4, lr=0.1, trainer_params=dict(max_epochs=2, gpus=0), train_batch_size=64
    )
    assert (
        repr(model) == "DeepARModel(decoder_length = 3, encoder_length = 4, "
        "dataset_builder = PytorchForecastingDatasetBuilder(max_encoder_length = 4, min_encoder_length = 4, "
        "min_prediction_idx = None, min_prediction_length = None, max_prediction_length = 3, static_categoricals = [], "
        "static_reals = [], time_varying_known_categoricals = [], time_varying_known_reals = ['time_idx'], "
        "time_varying_unknown_categoricals = [], time_varying_unknown_reals = ['target'], variable_groups = {}, "
        "constant_fill_strategy = [], allow_missing_timesteps = True, lags = {}, add_relative_time_idx = True, "
        "add_target_scales = True, add_encoder_length = True, target_normalizer = GroupNormalizer(groups=['segment']), "
        "categorical_encoders = {}, scalers = {}, ), "
        "train_batch_size = 64, test_batch_size = 64, lr = 0.1, cell_type = 'LSTM', hidden_size = 10, rnn_layers = 2, "
        "dropout = 0.1, loss = NormalDistributionLoss(), trainer_params = {'max_epochs': 2, 'gpus': 0}, quantiles_kwargs = {}, )"
    )


def test_deepar_forecast_throw_error_on_return_components():
    with pytest.raises(NotImplementedError, match="This mode isn't currently implemented!"):
        DeepARModel.forecast(self=Mock(), ts=Mock(), prediction_size=Mock(), return_components=True)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = DeepARModel(decoder_length=3, encoder_length=4, trainer_params=dict(max_epochs=1, gpus=0))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
