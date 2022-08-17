from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from etna.metrics import MAE
from etna.models.nn import MLPModel
from etna.models.nn.mlp import MLPNet
from etna.transforms import LagTransform
from etna.transforms import StandardScalerTransform


@pytest.mark.parametrize("horizon", [8, 13])
def test_mlp_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))
    std = StandardScalerTransform(in_column="target")
    ts_train.fit_transform([std, lag])

    decoder_length = 14
    model = MLPModel(
        input_size=3,
        encoder_length=0,
        hidden_size=[16, 10],
        decoder_length=decoder_length,
        trainer_params=dict(max_epochs=100),
    )
    future = ts_train.make_future(decoder_length)
    model.fit(ts_train)
    future = model.forecast(future, horizon=horizon)

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.7


def test_mlp_make_samples(example_df):
    mlp_module = MagicMock()
    encoder_length = 0
    decoder_length = 4

    ts_samples = list(
        MLPNet.make_samples(mlp_module, df=example_df, encoder_length=encoder_length, decoder_length=decoder_length)
    )
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]

    assert first_sample["segment"] == "segment_1"
    assert first_sample["decoder_real"].shape == (decoder_length, 0)
    assert first_sample["decoder_target"].shape == (decoder_length, 1)
    np.testing.assert_equal(example_df[["target"]].iloc[:decoder_length], first_sample["decoder_target"])
    np.testing.assert_equal(example_df[["target"]].iloc[1 : decoder_length + 1], second_sample["decoder_target"])


def test_mlp_step():
    torch.manual_seed(42)
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=None, optimizer_params=None)
    batch = {"decoder_real": torch.Tensor([1, 2, 3]), "decoder_target": torch.Tensor([1, 2, 3]), "segment": "A"}
    loss, decoder_target, _ = model.step(batch)
    assert round(float(loss.detach().numpy()), 2) == 5.21
    assert torch.all(decoder_target == torch.Tensor([1, 2, 3]))


def test_mlp_forward():
    torch.manual_seed(42)
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=None, optimizer_params=None)
    batch = {"decoder_real": torch.Tensor([1, 2, 3]), "decoder_target": torch.Tensor([1, 2, 3]), "segment": "A"}
    output = model.forward(batch)
    assert round(float(output.detach().numpy()), 2) == -0.13
