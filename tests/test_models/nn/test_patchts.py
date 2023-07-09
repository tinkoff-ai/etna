import pytest

from etna.metrics import MAE
from etna.models.nn import PatchTSModel
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.long_2
@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15
    ],
)
def test_patchts_model_run_weekly_overfit_with_scaler_small_patch(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    std = StandardScalerTransform(in_column="target")
    ts_train.fit_transform([std])
    encoder_length = 14
    decoder_length = 14
    model = PatchTSModel(
        encoder_length=encoder_length, decoder_length=decoder_length, patch_len=1, trainer_params=dict(max_epochs=20)
    )
    future = ts_train.make_future(horizon, transforms=[std], tail_steps=encoder_length)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)
    future.inverse_transform([std])

    mae = MAE("macro")
    print(mae(ts_test, future))
    assert mae(ts_test, future) < 0.9


@pytest.mark.long_2
@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15
    ],
)
def test_patchts_model_run_weekly_overfit_with_scaler_medium_patch(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    std = StandardScalerTransform(in_column="target")
    ts_train.fit_transform([std])
    encoder_length = 14
    decoder_length = 14
    model = PatchTSModel(
        encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=20)
    )
    future = ts_train.make_future(horizon, transforms=[std], tail_steps=encoder_length)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)
    future.inverse_transform([std])

    mae = MAE("macro")
    assert mae(ts_test, future) < 1.3


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = PatchTSModel(encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
