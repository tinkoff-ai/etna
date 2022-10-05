import pandas as pd
import pytest
from pytorch_forecasting.data import GroupNormalizer

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import DeepARModel
from etna.models.nn.utils import PytorchForecastingDatasetBuilder
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original


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
    ts_pred = ts_train.make_future(horizon, [dft], encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred, horizon=horizon)

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
    ts_pred = ts_train.make_future(horizon, [std, dft], encoder_length)
    model.fit(ts_train)
    ts_pred = model.forecast(ts_pred, horizon=horizon)
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
    future = example_tsds.make_future(horizon, tail_steps=horizon)
    forecast = model.forecast(future, horizon=horizon, prediction_interval=True, quantiles=[0.025, 0.975])
    for segment in forecast.segments:
        segment_slice = forecast[:, segment, :][segment]
        assert {"target_0.025", "target_0.975", "target"}.issubset(segment_slice.columns)
        assert (segment_slice["target_0.975"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target"] - segment_slice["target_0.025"] >= 0).all()
        assert (segment_slice["target_0.975"] - segment_slice["target"] >= 0).all()


def test_save_load(example_tsds):
    horizon = 3
    model = DeepARModel(max_epochs=2, learning_rate=[0.01], gpus=0, batch_size=64)
    transform = PytorchForecastingTransform(
        max_encoder_length=horizon,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["segment"]),
    )
    transforms = [transform]
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=transforms, horizon=horizon)
