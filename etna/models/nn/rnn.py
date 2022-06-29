from typing import Optional
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from etna.datasets import TSDataset
from etna.models.base import DeepBaseModel


class Batch(TypedDict):
    encoder_real: torch.Tensor
    decoder_real: torch.Tensor
    target: torch.Tensor


class RNN(LightningModule, DeepBaseModel):
    def __init__(
        self,
        input_size: int,
        decoder_length: int,
        encoder_length: int,
        num_layers: int = 2,
        hidden_size: int = 16,
        train_batch_size: int = 16,
        test_batch_size: int = 1,
        lr: float = 1e-3,
        loss=nn.MSELoss(),
        trainer_kwargs: dict = {},
        train_dataloader_kwargs: dict = {},
        test_dataloader_kwargs: dict = {},
        val_dataloader_kwargs: dict = {},
        split_kwargs: Optional[dict] = None,
        optimizer_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.decoder_length = decoder_length
        self.encoder_length = encoder_length
        self.loss = loss
        self.layer = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.hidden_size, input_size=self.input_size, batch_first=True
        )
        self.projection = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.trainer_kwargs = trainer_kwargs
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.test_dataloader_kwargs = test_dataloader_kwargs
        self.split_params = {} if split_kwargs is None else split_kwargs
        self.lr = lr
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

    def forward(self, x):
        encoder_real = x["encoder_real"].float()  # (batch_size, encoder_length-1, input_size)
        decoder_real = x["decoder_real"].float()  # (batch_size, decoder_real, input_size)
        decoder_length = decoder_real.shape[1]
        output, (h_n, c_n) = self.layer(encoder_real)
        forecast = torch.zeros(size=(decoder_real.shape[0], decoder_real.shape[1], 1)).float().to(decoder_real.device)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        return optimizer

    def training_step(self, batch: Batch, *args, **kwargs):
        encoder_real = batch["encoder_real"].float()  # (batch_size, encoder_lenght-1, input_size)
        decoder_real = batch["decoder_real"].float()  # (batch_size, decoder_length, input_size)
        target = batch["target"].float()

        decoder_length = decoder_real.shape[1]

        output, (_, _) = self.layer(torch.cat((encoder_real, decoder_real), dim=1))

        target_prediction = output[:, -decoder_length:]
        target_prediction = self.projection(target_prediction)

        target = target[:, -decoder_length:]

        return self.loss(target_prediction, target)

    def make_samples(self, x: pd.DataFrame) -> dict:

        encoder_length = self.encoder_length
        decoder_length = self.decoder_length

        def _make(x, start_idx, encoder_length, decoder_length) -> Optional[dict]:
            x_dict = {"target": list(), "encoder_real": list(), "decoder_real": list(), "segment": None}
            total_length = len(x["target"])
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return

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
            x_dict["target"] = (
                x["target"].values[start_idx : start_idx + decoder_length + encoder_length].reshape(-1, 1)
            )

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
