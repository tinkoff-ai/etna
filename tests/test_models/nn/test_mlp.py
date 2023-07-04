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
from tests.test_models.utils import assert_model_equals_loaded_original
from tests.test_models.utils import assert_sampling_is_valid


@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15,
    ],
)
def test_mlp_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 4)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    transforms = [std, lag, fourier]
    ts_train.fit_transform(transforms)

    decoder_length = 14
    model = MLPModel(
        input_size=10,
        hidden_size=[10, 10, 10, 10, 10],
        lr=1e-1,
        decoder_length=decoder_length,
        trainer_params=dict(max_epochs=100),
    )
    future = ts_train.make_future(horizon, transforms=transforms)
    model.fit(ts_train)
    future = model.forecast(future, prediction_size=horizon)
    future.inverse_transform(transforms)

    mae = MAE("macro")
    assert mae(ts_test, future) < 0.05


def test_mlp_make_samples(simple_df_relevance):
    mlp_module = MagicMock()
    df, df_exog = simple_df_relevance

    ts = TSDataset(df=df, df_exog=df_exog, freq="D")
    df = ts.to_flatten(ts.df)
    encoder_length = 0
    decoder_length = 5
    ts_samples = list(
        MLPNet.make_samples(
            mlp_module, df=df[df.segment == "1"], encoder_length=encoder_length, decoder_length=decoder_length
        )
    )
    first_sample = ts_samples[0]
    second_sample = ts_samples[1]
    last_sample = ts_samples[-1]
    expected = {
        "decoder_real": np.array([[58.0, 0], [59.0, 0], [60.0, 0], [61.0, 0], [62.0, 0]]),
        "decoder_target": np.array([[27.0], [28.0], [29.0], [30.0], [31.0]]),
        "segment": "1",
    }

    assert first_sample["segment"] == "1"
    assert first_sample["decoder_real"].shape == (decoder_length, 2)
    assert first_sample["decoder_target"].shape == (decoder_length, 1)
    assert len(ts_samples) == 7
    assert np.all(last_sample["decoder_target"] == expected["decoder_target"])
    assert np.all(last_sample["decoder_real"] == expected["decoder_real"])
    assert last_sample["segment"] == expected["segment"]
    np.testing.assert_equal(df[["target"]].iloc[:decoder_length], first_sample["decoder_target"])
    np.testing.assert_equal(df[["target"]].iloc[decoder_length : 2 * decoder_length], second_sample["decoder_target"])


def test_mlp_forward_fail_nans():
    batch = {
        "decoder_real": torch.Tensor([[torch.nan, 2, 3], [1, 2, 3], [1, 2, 3]]),
        "decoder_target": torch.Tensor([[1], [2], [3]]),
        "segment": "A",
    }
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=nn.MSELoss(), optimizer_params=None)
    with pytest.raises(ValueError, match="There are NaNs in features"):
        _ = model.forward(batch)


def test_mlp_step():

    batch = {
        "decoder_real": torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        "decoder_target": torch.Tensor([[1], [2], [3]]),
        "segment": "A",
    }
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=nn.MSELoss(), optimizer_params=None)
    loss, decoder_target, output = model.step(batch)
    assert type(loss) == torch.Tensor
    assert type(decoder_target) == torch.Tensor
    assert torch.all(decoder_target == batch["decoder_target"])
    assert type(output) == torch.Tensor
    assert output.shape == torch.Size([3, 1])


def test_mlp_step_fail_nans():
    batch = {
        "decoder_real": torch.Tensor([[torch.nan, 2, 3], [1, 2, 3], [1, 2, 3]]),
        "decoder_target": torch.Tensor([[1], [2], [3]]),
        "segment": "A",
    }
    model = MLPNet(input_size=3, hidden_size=[1], lr=1e-2, loss=nn.MSELoss(), optimizer_params=None)
    with pytest.raises(ValueError, match="There are NaNs in features"):
        _ = model.step(batch)


def test_mlp_layers():
    model = MLPNet(input_size=3, hidden_size=[10], lr=1e-2, loss=None, optimizer_params=None)
    model_ = nn.Sequential(
        nn.Linear(in_features=3, out_features=10), nn.ReLU(), nn.Linear(in_features=10, out_features=1)
    )
    assert repr(model_) == repr(model.mlp)


def test_save_load(example_tsds):
    horizon = 3
    model = MLPModel(
        input_size=9,
        hidden_size=[10],
        lr=1e-1,
        decoder_length=14,
        trainer_params=dict(max_epochs=1),
    )
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    transforms = [lag, fourier, std]
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=transforms, horizon=horizon)


@pytest.mark.parametrize(
    "model",
    [
        MLPModel(
            input_size=9,
            hidden_size=[5],
            lr=1e-1,
            decoder_length=14,
            trainer_params=dict(max_epochs=1),
        ),
        MLPModel(
            input_size=9,
            hidden_size=[5, 5],
            lr=1e-1,
            decoder_length=14,
            trainer_params=dict(max_epochs=1),
        ),
        MLPModel(
            input_size=9,
            hidden_size=[5, 5, 5],
            lr=1e-1,
            decoder_length=14,
            trainer_params=dict(max_epochs=1),
        ),
    ],
)
def test_params_to_tune(model, example_tsds):
    ts = example_tsds
    horizon = 3
    lag = LagTransform(in_column="target", lags=list(range(horizon, horizon + 3)))
    fourier = FourierTransform(period=7, order=3)
    std = StandardScalerTransform(in_column="target")
    transforms = [lag, fourier, std]
    ts.fit_transform(transforms)
    assert len(model.params_to_tune()) > 0
    assert_sampling_is_valid(model=model, ts=ts)
