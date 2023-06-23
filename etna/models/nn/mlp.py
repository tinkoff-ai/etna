from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
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


class MLPBatch(TypedDict):
    """Batch specification for MLP."""

    decoder_real: "torch.Tensor"
    decoder_target: "torch.Tensor"
    segment: "torch.Tensor"


class MLPNet(DeepBaseNet):
    """MLP model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int],
        lr: float,
        loss: "torch.nn.Module",
        optimizer_params: Optional[dict],
    ) -> None:
        """Init MLP model.

        Parameters
        ----------
        input_size:
            size of the input feature space: target plus extra features
        hidden_size:
            list of sizes of the hidden states
        lr:
            learning rate
        loss:
            loss function
        optimizer_params:
            parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.loss = loss
        self.optimizer_params = {} if optimizer_params is None else optimizer_params
        layers = [nn.Linear(in_features=input_size, out_features=hidden_size[0]), nn.ReLU()]
        for i in range(1, len(hidden_size)):
            layers.append(nn.Linear(in_features=hidden_size[i - 1], out_features=hidden_size[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=hidden_size[-1], out_features=1))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _validate_batch(batch: MLPBatch):
        if batch["decoder_real"].isnan().sum().item():
            raise ValueError("There are NaNs in features, this model can't work with them!")

    def forward(self, batch: MLPBatch):  # type: ignore
        """Forward pass.

        Parameters
        ----------
        batch:
            batch of data
        Returns
        -------
        :
            forecast
        """
        self._validate_batch(batch)
        decoder_real = batch["decoder_real"].float()
        return self.mlp(decoder_real)

    def step(self, batch: MLPBatch, *args, **kwargs):  # type: ignore
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
        self._validate_batch(batch)
        decoder_real = batch["decoder_real"].float()
        decoder_target = batch["decoder_target"].float()

        output = self.mlp(decoder_real)
        loss = self.loss(output, decoder_target)
        return loss, decoder_target, output

    def make_samples(self, df: pd.DataFrame, encoder_length: int, decoder_length: int) -> Iterable[dict]:
        """Make samples from segment DataFrame."""
        values_real = (
            df.select_dtypes(include=[np.number]).pipe(lambda x: x[[i for i in x.columns if i != "target"]]).values
        )
        values_target = df["target"].values
        segment = df["segment"].values[0]

        def _make(
            values_target: np.ndarray, values_real: np.ndarray, segment: str, start_idx: int, decoder_length: int
        ) -> Optional[dict]:

            sample: Dict[str, Any] = {"decoder_real": list(), "decoder_target": list(), "segment": None}
            total_length = len(df["target"])
            total_sample_length = decoder_length

            if total_sample_length + start_idx > total_length:
                return None

            sample["decoder_real"] = values_real[start_idx : start_idx + decoder_length]
            sample["decoder_target"] = values_target[start_idx : start_idx + decoder_length].reshape(-1, 1)
            sample["segment"] = segment
            return sample

        start_idx = 0
        while True:
            batch = _make(
                values_target=values_target,
                values_real=values_real,
                segment=segment,
                start_idx=start_idx,
                decoder_length=decoder_length,
            )
            if batch is None:
                break
            yield batch
            start_idx += decoder_length
        if start_idx < len(df):
            resid_length = len(df) - decoder_length
            batch = _make(
                values_target=values_target,
                values_real=values_real,
                segment=segment,
                start_idx=resid_length,
                decoder_length=decoder_length,
            )
            if batch is not None:
                yield batch

    def configure_optimizers(self):
        """Optimizer configuration."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        return optimizer


class MLPModel(DeepBaseModel):
    """MLPModel."""

    def __init__(
        self,
        input_size: int,
        decoder_length: int,
        hidden_size: List,
        encoder_length: int = 0,
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
        """Init MLP model.

        Parameters
        ----------
        input_size:
            size of the input feature space: target plus extra features
        decoder_length:
            decoder length
        hidden_size:
            List of sizes of the hidden states
        encoder_length:
            encoder length
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
            Pytorch ligthning trainer parameters (api reference :py:class:`pytorch_lightning.trainer.trainer.Trainer`)
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
        self.hidden_size = hidden_size
        self.lr = lr
        self.loss = loss
        self.optimizer_params = optimizer_params
        super().__init__(
            net=MLPNet(
                input_size=input_size,
                hidden_size=hidden_size,
                lr=lr,
                loss=nn.MSELoss() if loss is None else loss,
                optimizer_params=optimizer_params,
            ),
            encoder_length=encoder_length,
            decoder_length=decoder_length,
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

        This grid tunes parameters: ``lr``, ``hidden_size.i`` where i from 0 to ``len(hidden_size) - 1``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        grid: Dict[str, BaseDistribution] = {}

        for i in range(len(self.hidden_size)):
            key = f"hidden_size.{i}"
            value = IntDistribution(low=4, high=64, step=4)
            grid[key] = value

        grid["lr"] = FloatDistribution(low=1e-5, high=1e-2, log=True)
        return grid
