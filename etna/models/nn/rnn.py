from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import TypedDict

import numpy as np
import pandas as pd

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn

from etna.models.base import DeepBaseModel


class Batch(TypedDict):
    """Batch specification for RNN."""

    encoder_real: "torch.Tensor"
    decoder_real: "torch.Tensor"
    encoder_target: "torch.Tensor"
    decoder_target: "torch.Tensor"
    segment: "torch.Tensor"


class RNN(DeepBaseModel):
    """RNN based on LSTM cell."""

    def __init__(
        self,
        input_size: int,
        decoder_length: int,
        encoder_length: int,
        num_layers: int = 2,
        hidden_size: int = 16,
        lr: float = 1e-3,
        loss: Optional["torch.nn.Module"] = None,
        train_batch_size: int = 16,
        test_batch_size: int = 1,
        trainer_kwargs: Optional[dict] = None,
        train_dataloader_kwargs: Optional[dict] = None,
        test_dataloader_kwargs: Optional[dict] = None,
        val_dataloader_kwargs: Optional[dict] = None,
        split_kwargs: Optional[dict] = None,
        optimizer_kwargs: Optional[dict] = None,
    ) -> None:
        """Init RNN based on LSTM cell.

        Parameters
        ----------
        input_size:
            size of the input feature space: target plus extra features
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        num_layers:
            number of layers
        hidden_size:
            size of the hidden state
        lr:
            learning rate
        loss:
            loss function
        train_batch_size:
            batch size for training
        test_batch_size:
            batch size for testing
        trainer_kwargs:
            Pytorch ligthning trainer kwargs
        train_dataloader_kwargs:
            kwargs for train dataloader like sampler for example
        test_dataloader_kwargs:
            kwargs for test dataloader
        val_dataloader_kwargs:
            kwargs for validation dataloader
        split_kwargs:
            kwargs for torch dataset split for train-test splitting
        optimizer_kwargs:
            kwargs for optimizer
        """
        super().__init__(
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_dataloader_kwargs={} if train_dataloader_kwargs is None else train_dataloader_kwargs,
            test_dataloader_kwargs={} if test_dataloader_kwargs is None else test_dataloader_kwargs,
            val_dataloader_kwargs={} if val_dataloader_kwargs is None else val_dataloader_kwargs,
            trainer_kwargs={} if trainer_kwargs is None else trainer_kwargs,
            split_kwargs={} if split_kwargs is None else split_kwargs,
        )
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.loss = torch.nn.MSELoss() if loss is None else loss
        self.layer = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.hidden_size, input_size=self.input_size, batch_first=True
        )
        self.projection = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.lr = lr
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    def forward(self, x: Batch, *args, **kwargs):  # type: ignore
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
        encoder_real = x["encoder_real"].float()  # (batch_size, encoder_length-1, input_size)
        decoder_real = x["decoder_real"].float()  # (batch_size, decoder_length, input_size)
        decoder_length = decoder_real.shape[1]
        output, (h_n, c_n) = self.layer(encoder_real)
        forecast = torch.zeros(
            size=(decoder_real.shape[0], decoder_real.shape[1], 1), dtype=decoder_real.dtype, device=decoder_real.device
        )  # (batch_size, decoder_length, 1)

        for i in range(decoder_length - 1):
            output, (h_n, c_n) = self.layer(decoder_real[:, i, None], (h_n, c_n))
            forecast_point = self.projection(output[:, -1]).flatten()
            forecast[:, i, 0] = forecast_point
            decoder_real[:, i + 1, 0] = forecast_point

        output, (h_n, c_n) = self.layer(decoder_real[:, decoder_length - 1, None], (h_n, c_n))
        forecast_point = self.projection(output[:, -1]).flatten()
        forecast[:, decoder_length - 1, 0] = forecast_point
        return forecast

    def configure_optimizers(self):
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        return optimizer

    def step(self, batch: Batch, *args, **kwargs):  # type: ignore
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
        encoder_real = batch["encoder_real"].float()  # (batch_size, encoder_length-1, input_size)
        decoder_real = batch["decoder_real"].float()  # (batch_size, decoder_length, input_size)

        encoder_target = batch["encoder_target"].float()  # (batch_size, encoder_length-1, 1)
        decoder_target = batch["decoder_target"].float()  # (batch_size, decoder_length, 1)

        decoder_length = decoder_real.shape[1]

        output, (_, _) = self.layer(torch.cat((encoder_real, decoder_real), dim=1))

        target_prediction = output[:, -decoder_length:]
        target_prediction = self.projection(target_prediction)  # (batch_size, decoder_length, 1)

        target = decoder_target

        loss = self.loss(target_prediction, target)
        return loss, target, target_prediction

    def make_samples(self, x: pd.DataFrame) -> Iterator[dict]:
        """Make samples from segment DataFrame."""
        encoder_length = self.encoder_length
        decoder_length = self.decoder_length

        def _make(x, start_idx, encoder_length, decoder_length) -> Optional[dict]:
            x_dict: Dict[str, Any] = {"target": list(), "encoder_real": list(), "decoder_real": list(), "segment": None}
            total_length = len(x["target"])
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            x_dict["decoder_real"] = (
                x.select_dtypes(include=[np.number])
                .pipe(lambda x: x[["target"] + [i for i in x.columns if i != "target"]])
                .values[start_idx + encoder_length : start_idx + decoder_length + encoder_length]
            )
            x_dict["decoder_real"][:, 0] = (
                x["target"].shift(1).values[start_idx + encoder_length : start_idx + decoder_length + encoder_length]
            )
            x_dict["encoder_real"] = (
                x.select_dtypes(include=[np.number])
                .pipe(lambda x: x[["target"] + [i for i in x.columns if i != "target"]])
                .values[start_idx : start_idx + encoder_length]
            )
            x_dict["encoder_real"][:, 0] = x["target"].shift(1).values[start_idx : start_idx + encoder_length]
            target = x["target"].values[start_idx : start_idx + decoder_length + encoder_length].reshape(-1, 1)

            x_dict["encoder_target"] = target[1:encoder_length]
            x_dict["decoder_target"] = target[encoder_length:]

            x_dict["encoder_real"] = x_dict["encoder_real"][1:]
            x_dict["segment"] = x["segment"].values[0]

            return x_dict

        start_idx = 0
        while True:
            batch = _make(
                x=x,
                start_idx=start_idx,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
            )
            if batch is None:
                break
            yield batch
            start_idx += 1
