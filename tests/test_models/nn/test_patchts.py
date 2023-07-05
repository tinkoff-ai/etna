import pytest

from etna.metrics import MAE
from etna.models.nn import PatchTSModel
from etna.transforms import StandardScalerTransform


@pytest.mark.long_2
@pytest.mark.parametrize(
    "horizon",
    [
        8
    ],
)
def test_rnn_model_run_weekly_overfit_without_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    encoder_length = 14
    decoder_length = 14
    model = PatchTSModel(
        encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=10)
    )
    future = ts_train.make_future(horizon, tail_steps=encoder_length)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.06
