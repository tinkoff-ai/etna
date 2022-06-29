from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from typing_extensions import TypedDict

from etna.core import BaseMixin
from etna.models.base import DeepBaseModel


class TrainBatch(TypedDict):
    encoder_real: Tensor  # (batch_size, seq_length, input_size)
    datetime_index: Tensor  # (batch_size, seq_length, num_models)
    target: Tensor  # (batch_size, seq_length, 1)


class InferenceBatch(TypedDict):
    encoder_real: Tensor  # (batch_size, seq_length, input_size)
    decoder_real: Tensor  # (batch_size, horizon, input_size)
    datetime_index: Tensor  # (batch_size, seq_length + horizon, num_models)
    target: Tensor  # (batch_size, seq_length, 1)


class SSM(ABC, BaseMixin):
    @abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def emission_coeff(self, datetime_index: Tensor) -> Tensor:  # (batch_size, seq_length, latent_dim)
        raise NotImplementedError

    @abstractmethod
    def transition_coeff(self) -> Tensor:  # (latent_dim, latent_dim)
        raise NotImplementedError

    @abstractmethod
    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:  # (batch_size, seq_length, latent_dim)
        raise NotImplementedError


class LevelSSM(SSM):
    def latent_dim(self) -> int:
        return 1

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
        emission_coeff = torch.ones(datetime_index.shape[0], datetime_index.shape[1], self.latent_dim())
        return emission_coeff

    def transition_coeff(self) -> Tensor:
        transition_coeff = torch.eye(self.latent_dim())
        return transition_coeff

    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:
        return self.emission_coeff(datetime_index)


class LevelTrendSSM(LevelSSM):
    def latent_dim(self) -> int:
        return 2

    def transition_coeff(self) -> Tensor:
        transition_coeff = torch.eye(self.latent_dim())
        transition_coeff[0, 1] = 1
        return transition_coeff


class SeasonalitySSM(LevelSSM):
    def __init__(self, num_seasons: int):
        self.num_seasons = num_seasons

    def latent_dim(self) -> int:
        return self.num_seasons

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
        emission_coeff = torch.nn.functional.one_hot(datetime_index.squeeze(-1), num_classes=self.latent_dim()).float()
        return emission_coeff


class CompositeSSM(SSM):
    def __init__(
        self,
        seasonal_ssms: List[SeasonalitySSM],
        add_level: bool = False,
        add_level_trend: bool = False,
    ):
        self.add_level = add_level
        self.add_level_trend = add_level_trend
        self.seasonal_ssms = seasonal_ssms
        self.nonseasonal_ssm = []
        if self.add_level:
            self.nonseasonal_ssm = [LevelSSM()]
        elif self.add_level_trend:
            self.nonseasonal_ssm = [LevelTrendSSM()]
        self.ssms = self.seasonal_ssms + self.nonseasonal_ssm

    def latent_dim(self) -> int:
        return sum([ssm.latent_dim() for ssm in self.ssms])

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
        emission_coeff = torch.cat([ssm.emission_coeff(datetime_index[i]) for i, ssm in enumerate(self.ssms)], dim=-1)
        return emission_coeff

    def transition_coeff(self) -> Tensor:
        transition_coeff = torch.block_diag(*[ssm.transition_coeff() for ssm in self.ssms])
        return transition_coeff

    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:
        return self.emission_coeff(datetime_index)


class DeepStateNetwork(LightningModule, DeepBaseModel):
    def __init__(self, ssm: SSM, n_samples: int, input_size: int, decoder_length: int, encoder_length: int, train_batch_size: int, test_batch_size, trainer_kwargs, num_layers: int = 1):
        super().__init__()
        self.ssm = ssm
        self.n_samples = n_samples
        self.latent_dim = self.ssm.latent_dim()
        self.input_size = input_size
        self.num_layers = num_layers
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.trainer_kwargs=dict(max_epochs=20)
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.trainer_kwargs = trainer_kwargs
        self.train_dataloader_kwargs = dict()
        self.val_dataloader_kwargs = dict()
        self.test_dataloader_kwargs = dict()
        self.split_params = {}
        self.optimizer_kwargs = {}

        self.RNN = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.latent_dim, input_size=self.input_size, batch_first=True
        )
        self.projectors = nn.ModuleDict(
            dict(
                prior_mean=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                prior_std=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                innovation=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                noise_std=nn.Linear(in_features=self.latent_dim, out_features=1),
                offset=nn.Linear(in_features=self.latent_dim, out_features=1),
            )
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_ssm_parameters(
        self,
        train_batch: TrainBatch,
    ):
        encoder_real = train_batch["encoder_real"].float()  # (batch_size, seq_length, input_size)
        output, (_, _) = self.RNN(encoder_real)  # (batch_size, seq_length, latent_dim)
        innovation_coeff = (self.projectors["innovation"](output),)  # (batch_size, seq_length, latent_dim)
        noise_std = (self.projectors["noise_std"](output),)  # (batch_size, seq_length, 1)
        prior_mean = (self.projectors["prior_mean"](output[:, 0]),)  # (batch_size, latent_dim)
        prior_std = (self.projectors["prior_std"](output[:, 0]),)  # (batch_size, latent_dim)
        offset = (self.projectors["offset"](output),)  # (batch_size, seq_length, 1)
        return prior_mean, prior_std, noise_std, innovation_coeff, offset

    def forward(self, inference_batch: InferenceBatch):
        encoder_real = inference_batch["encoder_real"].float()  # (batch_size, seq_length, input_size)
        targets = inference_batch["target"].float()[:, : encoder_real.shape[1]]  # (batch_size, seq_length, 1)
        decoder_real = inference_batch["decoder_real"].float()  # (batch_size, horizon, input_size)
        datetime_index_train = (
            inference_batch["datetime_index"].permute(2, 0, 1).to(torch.int64)[:, :, : encoder_real.shape[1]]
        )  # (num_models, batch_size, seq_length, 1)
        datetime_index_test = (
            inference_batch["datetime_index"].permute(2, 0, 1).to(torch.int64)[:, :, encoder_real.shape[1] :]
        )  # (num_models, batch_size, horizon, 1)

        output, (h_n, c_n) = self.RNN(encoder_real)  # (batch_size, seq_length, latent_dim)
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index_train),  # (batch_size, seq_length, latent_dim)
            transition_coeff=self.ssm.transition_coeff().type_as(targets),  # (latent_dim, latent_dim)
            innovation_coeff=self.ssm.innovation_coeff(datetime_index_train)
            * self.projectors["innovation"](output),  # (batch_size, seq_length, latent_dim)
            noise_std=self.projectors["noise_std"](output),  # (batch_size, seq_length, 1)
            prior_mean=self.projectors["prior_mean"](output[:, 0]),  # (batch_size, latent_dim)
            prior_std=self.projectors["prior_std"](output[:, 0]),  # (batch_size, latent_dim)
            prior_cov=None,
            offset=self.projectors["offset"](output),  # (batch_size, seq_length, 1)
            seq_length=output.shape[1],
            latent_dim=self.latent_dim,
        )
        _, prior_mean, prior_cov = lds.log_likelihood(targets=targets)

        output, (_, _) = self.RNN(decoder_real, (h_n, c_n))  # (batch_size, seq_length, latent_dim)
        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index_test),  # (batch_size, seq_length, latent_dim)
            transition_coeff=self.ssm.transition_coeff().type_as(targets),  # (latent_dim, latent_dim)
            innovation_coeff=self.ssm.innovation_coeff(datetime_index_test)
            * self.projectors["innovation"](output),  # (batch_size, seq_length, latent_dim)
            noise_std=self.projectors["noise_std"](output),  # (batch_size, seq_length, latent_dim)
            prior_mean=prior_mean,  # (batch_size, latent_dim)
            prior_std=None,  # (batch_size, latent_dim)
            prior_cov=prior_cov,
            offset=self.projectors["offset"](output),  # (batch_size, seq_length, 1)
            seq_length=output.shape[1],
            latent_dim=self.latent_dim,
        )

        forecast = torch.mean(lds.sample(n_samples=self.n_samples), dim=0).squeeze(-1)
        return forecast

    def training_step(self, batch: TrainBatch, batch_idx):
        encoder_real = batch["encoder_real"].float()  # (batch_size, seq_length, input_size)
        targets = batch["target"].float()  # (batch_size, seq_length, 1)
        datetime_index = (
            batch["datetime_index"].permute(2, 0, 1).to(torch.int64)
        )  # (num_models, batch_size, seq_length, 1)

        output, (_, _) = self.RNN(encoder_real)  # (batch_size, seq_length, latent_dim)

        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(datetime_index),  # (batch_size, seq_length, latent_dim)
            transition_coeff=self.ssm.transition_coeff().type_as(targets),  # (latent_dim, latent_dim)
            innovation_coeff=self.ssm.innovation_coeff(datetime_index)
            * self.projectors["innovation"](output),  # (batch_size, seq_length, latent_dim)
            noise_std=self.projectors["noise_std"](output),  # (batch_size, seq_length, 1)
            prior_mean=self.projectors["prior_mean"](output[:, 0]),  # (batch_size, latent_dim)
            prior_std=self.projectors["prior_std"](output[:, 0]),  # (batch_size, latent_dim)
            prior_cov=None,
            offset=self.projectors["offset"](output),  # (batch_size, seq_length, 1)
            seq_length=output.shape[1],
            latent_dim=self.latent_dim,
        )
        log_likelihood, _, _ = lds.log_likelihood(targets=targets)
        log_likelihood = torch.mean(torch.sum(log_likelihood, dim=1))
        return -log_likelihood

    def make_samples(self, x: pd.DataFrame):
        import torch

        encoder_length = self.encoder_length
        decoder_length = self.decoder_length
        datetime_index = "timestamp"
        def _make(x, start_idx, encoder_length, decoder_length) -> Optional[dict]:
            columns_to_add = ["segment_code"]
            x_dict = {
                "target": list(),
                "encoder_real": list(),
                "decoder_real": list(),
                "datetime_index": list(),
                "segment": None,
            }
            total_length = len(x["target"])
            total_sample_length = encoder_length + decoder_length

            if total_sample_length + start_idx > total_length:
                return

            x_dict["decoder_real"] = x[columns_to_add].values[
                start_idx + encoder_length : start_idx + decoder_length + encoder_length
            ]
            x_dict["encoder_real"] = x[columns_to_add].values[start_idx : start_idx + encoder_length]
            x_dict["target"] = x["target"].values[start_idx : start_idx + decoder_length + encoder_length].reshape(-1, 1)
            x_dict["datetime_index"] = (
                x[datetime_index].astype(int).values[start_idx : start_idx + decoder_length + encoder_length]
            )

            x_dict["target"] = torch.from_numpy(x_dict["target"])
            x_dict["decoder_real"] = torch.from_numpy(x_dict["decoder_real"])
            x_dict["encoder_real"] = torch.from_numpy(x_dict["encoder_real"])
            x_dict["datetime_index"] = torch.from_numpy(x_dict["datetime_index"] % 7).to(torch.int64)
            x_dict["segment"] = x["segment"].values[0]

            return x_dict

        segment_start = x["target"].first_valid_index()
        segment_len = x.index.max() - segment_start + 1
        segment_start -= x.index.min()
        n_samples_per_segment = segment_len // (encoder_length + decoder_length)
        start_idxs = np.random.randint(
            segment_start, max(segment_start + 1, segment_len - encoder_length - decoder_length), size=n_samples_per_segment
        )
        samples = []
        for start_idx in start_idxs:
            batch = _make(
                x=x,
                start_idx=start_idx,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
            )
            samples.append(batch)
        return samples

class LDS:
    """Implements Linear Dynamical System (LDS) as a distribution."""

    def __init__(
        self,
        emission_coeff: Tensor,  # (batch_size, seq_length, latent_dim)
        transition_coeff: Tensor,  # (latent_dim, latent_dim)
        innovation_coeff: Tensor,  # (batch_size, seq_length, latent_dim)
        noise_std: Tensor,  # (batch_size, seq_length, 1)
        prior_mean: Tensor,  # (batch_size, latent_dim)
        prior_std: Optional[Tensor],  # (batch_size, latent_dim)
        prior_cov: Optional[Tensor],  # (batch_size, latent_dim, latent_dim)
        offset: Tensor,  # (batch_size, seq_length, 1)
        seq_length: int,
        latent_dim: int,
    ):
        self.emission_coeff = emission_coeff
        self.transition_coeff = transition_coeff
        self.innovation_coeff = innovation_coeff
        self.noise_std = noise_std
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.prior_cov = (
            prior_cov if prior_cov is not None else torch.diag_embed(prior_std * prior_std)
        )  # (batch_size, latent_dim, latent_dim)
        self.offset = offset
        self.seq_length = seq_length
        self.latent_dim = latent_dim

    def kalman_filter_step(
        self,
        target: Tensor,  # (batch_size, 1)
        noise_std: Tensor,  # (batch_size, 1)
        prior_mean: Tensor,  # (batch_size, latent_dim)
        prior_cov: Tensor,  # (batch_size, latent_dim, latent_dim)
        emission_coeff: Tensor,  # (batch_size, latent_dim)
        offset: Tensor,  # (batch_size, 1)
    ):
        """
        One step of the Kalman filter.
        This function computes the filtered state (mean and covariance) given the
        linear system coefficients the prior state (mean and variance),
        as well as observations.
        Parameters
        ----------
        target
            Observations of the system output, shape (batch_size, output_dim)
        noise_std
            Standard deviation of the output noise, shape (batch_size, output_dim)
        prior_mean(mu_t)
            Prior mean of the latent state, shape (batch_size, latent_dim)
        prior_cov(P_t)
            Prior covariance of the latent state, shape
            (batch_size, latent_dim, latent_dim)
        emission_coeff(H)
            Emission coefficient, shape (batch_size, output_dim, latent_dim)

        Returns
        -------
        Tensor
            Log probability, shape (batch_size, 1)
        Tensor
            Filtered_mean, shape (batch_size, latent_dim, 1)
        Tensor
            Filtered_covariance, shape (batch_size, latent_dim, latent_dim)
        """
        emission_coeff = emission_coeff.unsqueeze(-1)

        # H * mu (batch_size, 1)
        target_mean = (emission_coeff.permute(0, 2, 1) @ prior_mean.unsqueeze(-1)).squeeze(-1)
        # print(target_mean.shape)
        # v (batch_size, 1)
        residual = target - target_mean - offset
        # print(residual.shape)
        # R (batch_size, 1, 1)
        noise_cov = torch.diag_embed(noise_std * noise_std)
        # print(noise_cov.shape)
        # F (batch_size, 1, 1)
        target_cov = emission_coeff.permute(0, 2, 1) @ prior_cov @ emission_coeff + noise_cov
        # print(target_cov.shape)
        # K (batch_size, latent_dim)
        kalman_gain = (prior_cov @ emission_coeff @ torch.inverse(target_cov)).squeeze(-1)
        # print(kalman_gain.shape)

        # mu = mu_t + K * v (batch_size, latent_dim)
        filtered_mean = prior_mean + (kalman_gain.unsqueeze(-1) @ residual.unsqueeze(-1)).squeeze(-1)
        # print(filtered_mean.shape)
        # P = (I - KH)P_t (batch_size, latent_dim, latent_dim)
        filtered_cov = (
            torch.eye(self.latent_dim).type_as(target) - kalman_gain.unsqueeze(-1) @ emission_coeff.permute(0, 2, 1)
        ) @ prior_cov
        # print(filtered_cov.shape)
        # log-likelihood (batch_size, 1)
        log_p = (
            Normal(target_mean.squeeze(-1), torch.sqrt(target_cov.squeeze(-1).squeeze(-1)))
            .log_prob(target.squeeze(-1))
            .unsqueeze(-1)
        )
        return log_p, filtered_mean, filtered_cov

    def kalman_filter(self, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs Kalman filtering given observations.

        Parameters
        ----------
        targets
            Observations, shape (batch_size, seq_length, 1)

        Returns
        -------
        Tensor
            Log probabilities, shape (seq_length, batch_size)
        Tensor
            Mean of p(l_T | l_{T-1}), where T is seq_length, with shape
            (batch_size, latent_dim)
        Tensor
            Covariance of p(l_T | l_{T-1}), where T is seq_length, with shape
            (batch_size, latent_dim, latent_dim)
        """
        log_p_seq = []
        mean = self.prior_mean
        cov = self.prior_cov
        for t in range(self.seq_length):
            log_p, filtered_mean, filtered_cov = self.kalman_filter_step(
                target=targets[:, t],
                noise_std=self.noise_std[:, t],
                prior_mean=mean,
                prior_cov=cov,
                emission_coeff=self.emission_coeff[:, t],
                offset=self.offset[:, t],
            )
            log_p_seq.append(log_p)
            mean = (self.transition_coeff @ filtered_mean.unsqueeze(-1)).squeeze(-1)
            cov = self.transition_coeff @ filtered_cov @ self.transition_coeff.T + self.innovation_coeff[
                :, t
            ].unsqueeze(-1) @ self.innovation_coeff[:, t].unsqueeze(-1).permute(0, 2, 1)

        return torch.cat(log_p_seq, dim=1), mean, cov

    def log_likelihood(self, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the log-likelihood of the target.

        Parameters
        ----------
        targets:
            Tensor with targets of shape (batch_size, seq_length, 1)

        Returns
        -------
        Tensor
            Tensor with log-likelihoods of targets of shape (batch_size, seq_length)
        """

        log_p, final_mean, final_cov = self.kalman_filter(targets=targets)

        return log_p, final_mean, final_cov

    def sample(self, n_samples: int) -> Tensor:

        batch_size = self.prior_mean.shape[0]

        # (n_samples, batch_size, seq_length, latent_dim)
        eps_latent = MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim)).sample(
            (n_samples, batch_size, self.seq_length)
        )

        # (n_samples, batch_size, seq_length, 1)
        eps_observation = Normal(0, 1).sample((n_samples, batch_size, self.seq_length))

        # (n_samples, batch_size, latent_dim)
        l_t = MultivariateNormal(self.prior_mean, self.prior_cov).sample((n_samples,))
        samples_seq = []
        for t in range(self.seq_length):
            z_t = (
                (self.emission_coeff[:, t].unsqueeze(-1).permute(0, 2, 1) @ l_t.unsqueeze(-1))
                + self.offset[:, t].unsqueeze(0).unsqueeze(-1)
                + (self.noise_std[:, t].unsqueeze(0) * eps_observation[:, :, t].unsqueeze(-1)).unsqueeze(-1)
            )
            l_t = (self.transition_coeff @ l_t.unsqueeze(-1)).squeeze(-1) + self.innovation_coeff[:, t].unsqueeze(
                0
            ) * eps_latent[:, :, t]
            samples_seq.append(z_t)

        # (n_samples, batch_size, seq_length, 1)
        samples = torch.cat(samples_seq, dim=2)
        return samples
