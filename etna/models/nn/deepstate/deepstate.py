from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import TypedDict

from etna.models.base import DeepBaseModel
from etna.models.base import DeepBaseNet
from etna.models.nn.deepstate.linear_dynamic_system import LDS
from etna.models.nn.deepstate.state_space_model import CompositeSSM


class DeepStateTrainBatch(TypedDict):
    encoder_real: Tensor  # (batch_size, seq_length, input_size)
    datetime_index: Tensor  # (batch_size, num_models , seq_length)
    target: Tensor  # (batch_size, seq_length, 1)


class DeepStateInferenceBatch(TypedDict):
    encoder_real: Tensor  # (batch_size, seq_length, input_size)
    decoder_real: Tensor  # (batch_size, horizon, input_size)
    datetime_index: Tensor  # (batch_size, num_models, seq_length + horizon)
    target: Tensor  # (batch_size, seq_length, 1)


class DeepStateNet(DeepBaseNet):
    """DeepState network."""

    def __init__(
        self,
        ssm: CompositeSSM,
        input_size: int,
        num_layers: int,
        n_samples: int,
        lr: float,
        optimizer_params: Optional[dict],
    ):
        """Create instance of DeepStateNet.

        Parameters
        ----------
        ssm:
            State Space Model of the system.
        input_size:
            Size of the input feature space: target plus extra features.
        num_layers:
            Number of layers in RNN.
        n_samples:
            Number of samples to use in predictions generation.
        lr:
            Learning rate.
        optimizer_params:
            Parameters for optimizer for Adam optimizer (api reference :py:class:`torch.optim.Adam`)
        """
        super().__init__()
        self.ssm = ssm
        self.input_size = input_size
        self.num_layers = num_layers
        self.n_samples = n_samples
        self.lr = lr
        self.optimizer_params = {} if optimizer_params is None else optimizer_params
        self.latent_dim = self.ssm.latent_dim()

        self.RNN = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.latent_dim, input_size=self.input_size, batch_first=True
        )
        self.projectors = nn.ModuleDict(
            dict(
                prior_mean=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                prior_std=nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim), nn.Softplus()
                ),
                innovation=nn.Sequential(
                    nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim), nn.Softplus()
                ),
                noise_std=nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=1), nn.Softplus()),
                offset=nn.Linear(in_features=self.latent_dim, out_features=1),
            )
        )

    def step(self, batch: DeepStateTrainBatch, *args, **kwargs):  # type: ignore
        encoder_real = batch["encoder_real"]  # (batch_size, seq_length, input_size)
        targets = batch["target"]  # (batch_size, seq_length, 1)
        datetime_index = batch["datetime_index"]  # (num_models, batch_size, seq_length)
        seq_length = targets.shape[1]

        output, (_, _) = self.RNN(encoder_real)  # (batch_size, seq_length, latent_dim)
        prior_std = self.projectors["prior_std"](output[:, 0])

        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index),
            transition_coeff=self.ssm.transition_coeff(datetime_index),
            innovation_coeff=self.ssm.innovation_coeff(datetime_index) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=self.projectors["prior_mean"](output[:, 0]),
            prior_cov=torch.diag_embed(prior_std * prior_std),
            offset=self.projectors["offset"](output),
            seq_length=seq_length,
            latent_dim=self.latent_dim,
        )
        log_likelihood = lds.log_likelihood(targets=targets)
        log_likelihood = torch.mean(torch.sum(log_likelihood, dim=1))
        return -log_likelihood

    def forward(self, x: DeepStateInferenceBatch, *args, **kwargs):  # type: ignore
        encoder_real = x["encoder_real"]  # (batch_size, seq_length, input_size)
        seq_length = encoder_real.shape[1]
        targets = x["target"][:, :seq_length]  # (batch_size, seq_length, 1)
        decoder_real = x["decoder_real"]  # (batch_size, horizon, input_size)
        datetime_index_train = x["datetime_index"].permute(1, 0, 2)[
            :, :, :seq_length
        ]  # (num_models, batch_size, seq_length)
        datetime_index_test = x["datetime_index"].permute(1, 0, 2)[
            :, :, seq_length:
        ]  # (num_models, batch_size, horizon)

        output, (h_n, c_n) = self.RNN(encoder_real)  # (batch_size, seq_length, latent_dim)
        prior_std = self.projectors["prior_std"](output[:, 0])
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index_train),
            transition_coeff=self.ssm.transition_coeff(datetime_index_train),
            innovation_coeff=self.ssm.innovation_coeff(datetime_index_train) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=self.projectors["prior_mean"](output[:, 0]),
            prior_cov=torch.diag_embed(prior_std * prior_std),
            offset=self.projectors["offset"](output),
            seq_length=seq_length,
            latent_dim=self.latent_dim,
        )
        _, prior_mean, prior_cov = lds.kalman_filter(targets=targets)

        output, (_, _) = self.RNN(decoder_real, (h_n, c_n))  # (batch_size, horizon, latent_dim)
        horizon = output.shape[1]
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index_test),
            transition_coeff=self.ssm.transition_coeff(datetime_index_test),
            innovation_coeff=self.ssm.innovation_coeff(datetime_index_test) * self.projectors["innovation"](output),
            noise_std=self.projectors["noise_std"](output),
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            offset=self.projectors["offset"](output),
            seq_length=horizon,
            latent_dim=self.latent_dim,
        )

        forecast = torch.mean(lds.sample(n_samples=self.n_samples), dim=0).squeeze(-1)
        return forecast

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

            sample["target"] = df["target"].values[start_idx : start_idx + total_sample_length].reshape(-1, 1)
            sample["datetime_index"] = self.ssm.generate_datetime_index(
                df["timestamp"].values[start_idx : start_idx + total_sample_length]
            )
            sample["segment"] = df["segment"].values[0]
            df = df.drop(columns=["target", "segment", "timestamp"])

            sample["encoder_real"] = df.values[start_idx : start_idx + encoder_length]
            sample["decoder_real"] = df.values[start_idx + encoder_length : start_idx + total_sample_length]

            sample["target"] = torch.from_numpy(sample["target"]).float()
            sample["decoder_real"] = torch.from_numpy(sample["decoder_real"]).float()
            sample["encoder_real"] = torch.from_numpy(sample["encoder_real"]).float()
            sample["datetime_index"] = torch.from_numpy(sample["datetime_index"]).to(torch.int64)
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


class DeepStateModel(DeepBaseModel):
    """DeepState model."""

    def __init__(
        self,
        ssm: CompositeSSM,
        input_size: int,
        encoder_length: int,
        decoder_length: int,
        num_layers: int = 1,
        n_samples: int = 5,
        lr: float = 1e-3,
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
        ssm:
            state Space Model of the system
        input_size:
            size of the input feature space: target plus extra features
        encoder_length:
            encoder length
        decoder_length:
            decoder length
        num_layers:
            number of layers in RNN
        n_samples:
            number of samples to use in predictions generation
        num_layers:
            number of layers
        lr:
            learning rate
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
        super().__init__(
            net=DeepStateNet(
                ssm=ssm,
                input_size=input_size,
                num_layers=num_layers,
                n_samples=n_samples,
                lr=lr,
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
