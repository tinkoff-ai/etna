from unittest.mock import MagicMock

import numpy as np
import pandas as pd
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
    assert mae(ts_test, future) < 1.7


@pytest.fixture()
def example_df_with_lag(random_seed):
    df1 = pd.DataFrame()
    df1["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df1["segment"] = "segment_1"
    df1["target"] = np.arange(len(df1)) + 2 * np.random.normal(size=len(df1))
    df1["lag"] = df1["target"].shift(1)

    df2 = pd.DataFrame()
    df2["timestamp"] = pd.date_range(start="2020-01-01", end="2020-02-01", freq="H")
    df2["segment"] = "segment_2"
    df2["target"] = np.sqrt(np.arange(len(df2)) + 2 * np.cos(np.arange(len(df2))))
    df2["lag"] = df2["target"].shift(1)

    return pd.concat([df1, df2], ignore_index=True)


def test_mlp_make_samples(example_df_with_lag):
    mlp_module = MagicMock()
    encoder_length = 0
    decoder_length = 4
    ts_samples = list(MLPNet.make_samples(mlp_module, df=example_df_with_lag, decoder_length=decoder_length))
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]

    assert first_sample["segment"] == "segment_1"
    assert first_sample["decoder_real"].shape == (decoder_length, 1)
    assert first_sample["decoder_target"].shape == (decoder_length, 1)
    np.testing.assert_equal(example_df_with_lag[["target"]].iloc[:decoder_length], first_sample["decoder_target"])
    np.testing.assert_equal(
        example_df_with_lag[["target"]].iloc[decoder_length : 2 * decoder_length], second_sample["decoder_target"]
    )


def test_mlp_step(random_seed):
    random_seed
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=None, optimizer_params=None)
    batch = {"decoder_real": torch.Tensor([1, 2, 3]), "decoder_target": torch.Tensor([1, 2, 3]), "segment": "A"}
    loss, decoder_target, output = model.step(batch)
    assert type(loss) == torch.Tensor
    assert type(decoder_target) == torch.Tensor
    assert torch.all(decoder_target == batch["decoder_target"])
    assert type(output) == torch.Tensor
    assert output.shape == torch.Size([1])


def test_mlp_forward():
    torch.manual_seed(42)
    mlp_module = MagicMock()
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=None, optimizer_params=None)
    batch = {"decoder_real": torch.Tensor([1, 2, 3]), "decoder_target": torch.Tensor([1, 2, 3]), "segment": "A"}
    MLPNet.forward(mlp_module, batch)
