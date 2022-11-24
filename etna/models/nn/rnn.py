from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import TypedDict

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch
    import torch.nn as nn

from etna.models.base import DeepBaseModel
from etna.models.base import DeepBaseNet


class RNNBatch(TypedDict):
    """Batch specification for RNN."""

    encoder_real: "torch.Tensor"
    decoder_real: "torch.Tensor"
    encoder_target: "torch.Tensor"
    decoder_target: "torch.Tensor"
    segment: "torch.Tensor"


class RNNNet(DeepBaseNet):
    """RNN based Lightning module with LSTM cell."""

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        lr: float,
        loss: "torch.nn.Module",
        optimizer_params: Optional[dict],
    ) -> None:
        """Init RNN based on LSTM cell.

        Parameters
        ----------
        input_size:
            size of the input feature space: target plus extra features
        num_layers:
            number of layers
        hidden_size:
            size of the hidden state
        lr:
            learning rate
        loss:
            loss function
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.loss = torch.nn.MSELoss() if loss is None else loss
        self.rnn = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.hidden_size, input_size=self.input_size, batch_first=True
        )
        self.projection = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.lr = lr
        self.optimizer_params = {} if optimizer_params is None else optimizer_params

    def forward(self, x: RNNBatch, *args, **kwargs):  # type: ignore
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
        decoder_target = x["decoder_target"].float()  # (batch_size, decoder_length, 1)
        decoder_length = decoder_real.shape[1]
        output, (h_n, c_n) = self.rnn(encoder_real)
        forecast = torch.zeros_like(decoder_target)  # (batch_size, decoder_length, 1)

        for i in range(decoder_length - 1):
            output, (h_n, c_n) = self.rnn(decoder_real[:, i, None], (h_n, c_n))
            forecast_point = self.projection(output[:, -1]).flatten()
            forecast[:, i, 0] = forecast_point
            decoder_real[:, i + 1, 0] = forecast_point

        # Last point is computed out of the loop because `decoder_real[:, i + 1, 0]` would cause index error
        output, (h_n, c_n) = self.rnn(decoder_real[:, decoder_length - 1, None], (h_n, c_n))
        forecast_point = self.projection(output[:, -1]).flatten()
        forecast[:, decoder_length - 1, 0] = forecast_point
        return forecast

    def step(self, batch: RNNBatch, *args, **kwargs):  # type: ignore
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

        output, (_, _) = self.rnn(torch.cat((encoder_real, decoder_real), dim=1))

        target_prediction = output[:, -decoder_length:]
        target_prediction = self.projection(target_prediction)  # (batch_size, decoder_length, 1)

        loss = self.loss(target_prediction, decoder_target)
        return loss, decoder_target, target_prediction

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterator[dict]:
        """Make samples from segment DataFrame."""

        def _make(df: pd.DataFrame, start_idx: int, encoder_length: int, decoder_length: int) -> Optional[dict]:
            sample: Dict[str, Any] = {
                "encoder_real": list(),
                "decoder_real": list(),
                "encoder_target": list(),
                "decoder_target": list(),
                "segment": None,
            }
            total_length = len(df["target"])
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            # Get shifted target and concatenate it with real values features
            sample["decoder_real"] = (
                df.select_dtypes(include=[np.number])
                .pipe(lambda x: x[["target"] + [i for i in x.columns if i != "target"]])
                .values[start_idx + encoder_length : start_idx + encoder_length + decoder_length]
            )
            sample["decoder_real"][:, 0] = (
                df["target"].shift(1).values[start_idx + encoder_length : start_idx + encoder_length + decoder_length]
            )

            # Get shifted target and concatenate it with real values features
            sample["encoder_real"] = (
                df.select_dtypes(include=[np.number])
                .pipe(lambda x: x[["target"] + [i for i in x.columns if i != "target"]])
                .values[start_idx : start_idx + encoder_length]
            )
            sample["encoder_real"][:, 0] = df["target"].shift(1).values[start_idx : start_idx + encoder_length]
            sample["encoder_real"] = sample["encoder_real"][1:]

            target = df["target"].values[start_idx : start_idx + encoder_length + decoder_length].reshape(-1, 1)
            sample["encoder_target"] = target[1:encoder_length]
            sample["decoder_target"] = target[encoder_length:]

            sample["segment"] = df["segment"].values[0]

            return sample

        start_idx = 0
        while True:
            batch = _make(
                df=df,
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


class RNNModel(DeepBaseModel):
    """RNN based model on LSTM cell."""

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
        test_batch_size: int = 16,
        optimizer_params: Optional[dict] = None,
        trainer_params: Optional[dict] = None,
        train_dataloader_params: Optional[dict] = None,
        test_dataloader_params: Optional[dict] = None,
        val_dataloader_params: Optional[dict] = None,
        split_params: Optional[dict] = None,
    ):
        """Init RNN model based on LSTM cell.

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
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params
        super().__init__(
            net=RNNNet(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                lr=lr,
                loss=nn.MSELoss() if loss is None else loss,
                optimizer_params=optimizer_params,
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
