from unittest.mock import MagicMock

import numpy as np
import pytest

from etna.metrics import MAE
from etna.models.nn import RNNModel
from etna.models.nn.rnn import RNNNet
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.long_2
@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15,
    ],
)
def test_rnn_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    """
    Given: I have dataframe with 2 segments with weekly seasonality with known future
    When: I use scale transformations
    Then: I get {horizon} periods per dataset as a forecast and they "the same" as past
    """

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    std = StandardScalerTransform(in_column="target")
    ts_train.fit_transform([std])

    encoder_length = 14
    decoder_length = 14
    model = RNNModel(
        input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=100)
    )
    future = ts_train.make_future(horizon, transforms=[std], tail_steps=encoder_length)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)
    future.inverse_transform([std])

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.06


def test_rnn_make_samples(example_df):
    rnn_module = MagicMock()
    encoder_length = 8
    decoder_length = 4

    ts_samples = list(
        RNNNet.make_samples(rnn_module, df=example_df, encoder_length=encoder_length, decoder_length=decoder_length)
    )
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]

    assert first_sample["segment"] == "segment_1"
    assert first_sample["encoder_real"].shape == (encoder_length - 1, 1)
    assert first_sample["decoder_real"].shape == (decoder_length, 1)
    assert first_sample["encoder_target"].shape == (encoder_length - 1, 1)
    assert first_sample["decoder_target"].shape == (decoder_length, 1)
    np.testing.assert_equal(example_df[["target"]].iloc[: encoder_length - 1], first_sample["encoder_real"])
    np.testing.assert_equal(example_df[["target"]].iloc[1:encoder_length], second_sample["encoder_real"])


@pytest.mark.parametrize("encoder_length", [1, 2, 10])
def test_context_size(encoder_length):
    encoder_length = encoder_length
    decoder_length = encoder_length
    model = RNNModel(
        input_size=1, encoder_length=encoder_length, decoder_length=decoder_length, trainer_params=dict(max_epochs=100)
    )

    assert model.context_size == encoder_length


def test_save_load(example_tsds):
    model = RNNModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[], horizon=3)


def test_params_to_tune(example_tsds):
    ts = example_tsds
    model = RNNModel(input_size=1, encoder_length=14, decoder_length=14, trainer_params=dict(max_epochs=1))
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
