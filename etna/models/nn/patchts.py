import math
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from etna import SETTINGS
from etna.distributions import BaseDistribution
from etna.distributions import FloatDistribution
from etna.distributions import IntDistribution
from etna.models.base import DeepBaseModel
from etna.models.base import DeepBaseNet

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn


class PatchTSBatch(TypedDict):
    """Batch specification for PatchTS."""

    encoder_real: "torch.Tensor"
    decoder_real: "torch.Tensor"
    encoder_target: "torch.Tensor"
    decoder_target: "torch.Tensor"
    segment: "torch.Tensor"


class PositionalEncoding(nn.Module):

    """Positional encoding of tokens and reshaping."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Tensor, shape [batch_size, input_size, patch_num, embedding_dim]."""
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x.shape == (batch_size * input_size, patch_num, embedding_dim)
        x = x.permute(1, 0, 2)  # (patch_num, batch_size * input_size, embedding_dim)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PatchTSNet(DeepBaseNet):
    """PatchTS based Lightning module."""

    def __init__(
        self,
        encoder_length: int,
        patch_len: int,
        stride: int,
        num_layers: int,
        hidden_size: int,
        feedforward_size: int,
        nhead: int,
        lr: float,
        loss: "torch.nn.Module",
        optimizer_params: Optional[dict],
    ) -> None:
        """Init PatchTS.

        Parameters
        ----------
        encoder_length:
            encoder length
        patch_len:
            size of patch
        stride:
            step of patch
        num_layers:
            number of layers
        hidden_size:
            size of the hidden state
        feedforward_size:
            size of feedforward layers in transformer
        nhead:
            number of transformer heads
        lr:
            learning rate
        loss:
            loss function
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.patch_len = patch_len
        self.num_layers = num_layers
        self.feedforward_size = feedforward_size
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.stride = stride
        self.loss = loss

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=self.nhead, dim_feedforward=self.feedforward_size
        )
        self.model = nn.Sequential(
            nn.Linear(self.patch_len, self.hidden_size),
            PositionalEncoding(d_model=self.hidden_size),
            nn.TransformerEncoder(encoder_layers, self.num_layers),
        )
        self.max_patch_num = (encoder_length - self.patch_len) // self.stride + 1
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=-2), nn.Linear(in_features=self.hidden_size * self.max_patch_num, out_features=1)
        )

        self.lr = lr
        self.optimizer_params = {} if optimizer_params is None else optimizer_params

    def forward(self, x: PatchTSBatch, *args, **kwargs):  # type: ignore
        """Forward pass.

        Parameters
        ----------
        x:
            batch of data

        Returns
        -------
        :
            forecast with shape (batch_size, decoder_length, 1)
        """
        encoder_real = x["encoder_real"].float()  # (batch_size, encoder_length, input_size)
        decoder_real = x["decoder_real"].float()  # (batch_size, decoder_length, input_size)
        decoder_length = decoder_real.shape[1]
        outputs = []
        current_input = encoder_real
        for _ in range(decoder_length):
            pred = self._get_prediction(current_input)
            outputs.append(pred)
            current_input = torch.cat((current_input[:, 1:, :], torch.unsqueeze(pred, dim=1)), dim=1)

        forecast = torch.cat(outputs, dim=1)
        forecast = torch.unsqueeze(forecast, dim=2)

        return forecast

    def _get_prediction(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch_size, input_size, encoder_length)
        # do patching
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # (batch_size, input_size, patch_num, patch_len)

        y = self.model(x)
        y = y.permute(1, 0, 2)  # (batch_size, hidden_size, patch_num)

        return self.projection(y)  # (batch_size, 1)

    def step(self, batch: PatchTSBatch, *args, **kwargs):  # type: ignore
        """Step for loss computation for training or validation.

        Parameters
        ----------
        batch:
            batch of data

        Returns
        -------
        :
            loss, true_target, prediction_target
        """
        encoder_real = batch["encoder_real"].float()  # (batch_size, encoder_length, input_size)
        decoder_real = batch["decoder_real"].float()  # (batch_size, decoder_length, input_size)

        decoder_target = batch["decoder_target"].float()  # (batch_size, decoder_length, 1)

        decoder_length = decoder_real.shape[1]

        outputs = []
        x = encoder_real
        for i in range(decoder_length):
            pred = self._get_prediction(x)
            outputs.append(pred)
            x = torch.cat((x[:, 1:, :], torch.unsqueeze(decoder_real[:, i, :], dim=1)), dim=1)

        target_prediction = torch.cat(outputs, dim=1)
        target_prediction = torch.unsqueeze(target_prediction, dim=2)

        loss = self.loss(target_prediction, decoder_target)
        return loss, decoder_target, target_prediction

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """Make samples from segment DataFrame."""
        values_real = df.select_dtypes(include=[np.number]).values
        values_target = df["target"].values
        segment = df["segment"].values[0]

        def _make(
            values_real: np.ndarray,
            values_target: np.ndarray,
            segment: str,
            start_idx: int,
            encoder_length: int,
            decoder_length: int,
        ) -> Optional[dict]:

            sample: Dict[str, Any] = {
                "encoder_real": list(),
                "decoder_real": list(),
                "encoder_target": list(),
                "decoder_target": list(),
                "segment": None,
            }
            total_length = len(values_target)
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            sample["decoder_real"] = values_real[start_idx + encoder_length : start_idx + total_sample_length]
            sample["encoder_real"] = values_real[start_idx : start_idx + encoder_length]

            target = values_target[start_idx : start_idx + encoder_length + decoder_length].reshape(-1, 1)
            sample["encoder_target"] = target[:encoder_length]
            sample["decoder_target"] = target[encoder_length:]

            sample["segment"] = segment

            return sample

        start_idx = 0
        while True:
            batch = _make(
                values_target=values_target,
                values_real=values_real,
                segment=segment,
                start_idx=start_idx,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
            )
            if batch is None:
                break
            yield batch
            start_idx += 1

    def configure_optimizers(self) -> "torch.optim.Optimizer":
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        return optimizer


class PatchTSModel(DeepBaseModel):
    """PatchTS model using PyTorch layers."""

    def __init__(
        self,
        decoder_length: int,
        encoder_length: int,
        patch_len: int = 4,
        stride: int = 1,
        num_layers: int = 3,
        hidden_size: int = 128,
        feedforward_size: int = 256,
        nhead: int = 16,
        lr: float = 1e-3,
        loss: Optional["torch.nn.Module"] = None,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        """Init PatchTS model.

        Parameters
        ----------
         encoder_length:
            encoder length
        decoder_length:
            decoder length
        patch_len:
            size of patch
        stride:
            step of patch
        num_layers:
            number of layers
        hidden_size:
            size of the hidden state
        feedforward_size:
            size of feedforward layers in transformer
        nhead:
            number of transformer heads
        lr:
            learning rate
        loss:
            loss function, MSELoss by default
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        trainer_params:
            Pytorch ligthning  trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`)
        train_dataloader_params:
            parameters for train dataloader like sampler for example (api reference :py:class:`torch.utils.data.DataLoader`)
        test_dataloader_params:
            parameters for test dataloader
        val_dataloader_params:
            parameters for validation dataloader
        split_params:
            dictionary with parameters for :py:func:`torch.utils.data.random_split` for train-test splitting
                * **train_size**: (*float*) value from 0 to 1 - fraction of samples to use for training

                * **generator**: (*Optional[torch.Generator]*) - generator for reproducibile train-test splitting

                * **torch_dataset_size**: (*Optional[int]*) - number of samples in dataset, in case of dataset not implementing ``__len__``
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.patch_len = patch_len
        self.stride = stride
        self.nhead = nhead
        self.feedforward_size = feedforward_size
        self.loss = loss if loss is not None else nn.MSELoss()
        self.optimizer_params = optimizer_params
        super().__init__(
            net=PatchTSNet(
                encoder_length,
                patch_len=self.patch_len,
                stride=self.stride,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                feedforward_size=self.feedforward_size,
                nhead=self.nhead,
                lr=self.lr,
                loss=self.loss,
                optimizer_params=self.optimizer_params,
            ),
            decoder_length=decoder_length,
            encoder_length=encoder_length,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_params=train_dataloader_params,
            test_dataloader_params=test_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            trainer_params=trainer_params,
            split_params=split_params,
        )

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``num_layers``, ``hidden_size``, ``lr``, ``encoder_length``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "num_layers": IntDistribution(low=1, high=3),
            "hidden_size": IntDistribution(low=16, high=256, step=self.nhead),
            "lr": FloatDistribution(low=1e-5, high=1e-2, log=True),
            "encoder_length": IntDistribution(low=self.patch_len, high=24),
        }
