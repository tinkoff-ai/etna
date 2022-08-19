from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from etna.datasets.tsdataset import TSDataset
from etna.metrics import MAE
from etna.models.nn import MLPModel
from etna.models.nn.mlp import MLPNet
from etna.transforms import FourierTransform
from etna.transforms import LagTransform
from etna.transforms import StandardScalerTransform


@pytest.mark.parametrize("horizon", [8, 13])
def test_mlp_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):

    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 4)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    ts_train.fit_transform([std, lag, fourier])

    decoder_length = 14
    model = MLPModel(
        input_size=10,
        hidden_size=[10, 10, 10, 10, 10],
        lr=1e-1,
        decoder_length=decoder_length,
        trainer_params=dict(max_epochs=100),
    )
    future = ts_train.make_future(decoder_length)
    model.fit(ts_train)
    future = model.forecast(future, horizon=horizon)

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.05


def test_mlp_make_samples(simple_df_relevance):
    mlp_module = MagicMock()
    df, df_exog = simple_df_relevance

    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    df = ts.to_flatten(ts.df)
    encoder_length = 0
    decoder_length = 4
    ts_samples = list(
        MLPNet.make_samples(mlp_module, df=df, encoder_length=encoder_length, decoder_length=decoder_length)
    )
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]

    assert first_sample["segment"] == "1"
    assert first_sample["decoder_real"].shape == (decoder_length, 2)
    assert first_sample["decoder_target"].shape == (decoder_length, 1)
    np.testing.assert_equal(df[["target"]].iloc[:decoder_length], first_sample["decoder_target"])
    np.testing.assert_equal(df[["target"]].iloc[decoder_length : 2 * decoder_length], second_sample["decoder_target"])


def test_mlp_step():
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=None, optimizer_params=None)
    batch = {
        "decoder_real": torch.Tensor([[1, 2, 3], [1, 2, 3]]),
        "decoder_target": torch.Tensor([1, 2, 3]),
        "segment": "A",
    }
    loss, decoder_target, output = model.step(batch)
    assert type(loss) == torch.Tensor
    assert type(decoder_target) == torch.Tensor
    assert torch.all(decoder_target == batch["decoder_target"])
    assert type(output) == torch.Tensor
    assert output.shape == torch.Size([2, 1])


def test_mlp_layers():
    model = MLPNet(input_size=3, hidden_size=[10], lr=1e-2, loss=None, optimizer_params=None)
    model_ = nn.Sequential(
        nn.Linear(in_features=3, out_features=10), nn.ReLU(), nn.Linear(in_features=10, out_features=1)
    )
    assert repr(model_) == repr(model.mlp)
