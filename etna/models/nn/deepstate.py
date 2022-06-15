from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import TypedDict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor


class TrainBatch(TypedDict):
    encoder_real: Tensor
    seasonal_indicator: Tensor
    target: Tensor


class InferenceBatch(TypedDict):
    encoder_real: Tensor


class SSM(ABC):
    @abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def emission_coeff(self, time_index: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def transition_coeff(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def innovation_coeff(self, time_index: Tensor) -> Tensor:
        raise NotImplementedError


class LevelSSM(SSM):
    def latent_dim(self) -> int:
        return 1

    def emission_coeff(self, time_index: Tensor) -> Tensor:  # (batch_size, seq_length)
        emission_coeff = torch.ones(time_index.shape[0], time_index.shape[1], self.latent_dim())
        return emission_coeff  # (batch_size, seq_length, latent_dim)

    def transition_coeff(self) -> Tensor:
        transition_coeff = torch.eye(self.latent_dim())
        return transition_coeff

    def innovation_coeff(self, time_index: Tensor) -> Tensor:  # (batch_size, seq_length)
        return self.emission_coeff(time_index)  # (batch_size, seq_length, latent_dim)


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

    def emission_coeff(self, time_index: Tensor) -> Tensor:  # (batch_size, seq_length)
        emission_coeff = (
            torch.nn.functional.one_hot(time_index, num_classes=self.latent_dim())
            .reshape(-1, self.latent_dim(), 1)
            .float()
        )
        return emission_coeff  # (batch_size, seq_length, latent_dim)

    def innovation_coeff(self, time_index: Tensor) -> Tensor:  # (batch_size, seq_length)
        return self.emission_coeff(time_index)  # (batch_size, seq_length, latent_dim)


class DeepStateNetwork(LightningModule):
    def __init__(self, ssm: SSM, input_size: int, seq_length: int, num_layers: int = 1):
        super(DeepStateNetwork).__init__()
        self.ssm = ssm
        self.latent_dim = self.ssm.latent_dim()
        self.input_size = input_size
        self.num_layers = num_layers
        self.seq_length = seq_length

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
            )
        )


    def training_step(self, train_batch: TrainBatch, batch_idx):
        encoder_real = train_batch["encoder_real"].float()  # (batch_size, encoder_length, input_size)
        targets = train_batch["target"].float()  # (batch_size, encoder_length)
        timestamps = train_batch["seasonal_indicator"]

        output, (_, _) = self.RNN(encoder_real)  # (batch_size, encoder_length, latent_dim)

        lds = LDS(
            emission_coeff=self.ssm.emission_coeff(timestamps),  # (batch_size, encoder_length, latent_dim, 1)
            transition_coeff=self.ssm.transition_coeff(),  # (latent_dim, latent_dim)
            innovation_coeff=self.ssm.innovation_coeff(timestamps),  # (batch_size, encoder_length, latent_dim, 1)
            noise_std=self.projectors["noise_std"](output),  # (batch_size, encoder_length, latent_dim, 1)
            prior_mean=self.projectors["prior_mean"](output[:, 0]),  # (batch_size, latent_dim, 1)
            prior_cov=self.projectors["prior_std"](output[:, 0]),  # (batch_size, latent_dim, latent_dim)
            seq_length=self.seq_length,
            latent_dim=self.latent_dim,
        )
        log_likelihood, _, _ = lds.log_likelihood(targets=targets)
        log_likelihood = torch.sum(log_likelihood, dim=0)
        return -log_likelihood


class LDS:
    """Implements Linear Dynamical System (LDS) as a distribution."""

    def __init__(
        self,
        emission_coeff: Tensor,
        transition_coeff: Tensor,
        innovation_coeff: Tensor,
        noise_std: Tensor,
        prior_mean: Tensor,
        prior_cov: Tensor,
        seq_length: int,
        latent_dim: int,
    ):
        self.emission_coeff = emission_coeff
        self.transition_coeff = transition_coeff
        self.innovation_coeff = innovation_coeff
        self.noise_std = noise_std
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.seq_length = seq_length
        self.latent_dim = latent_dim

    def kalman_filter_step(
        self,
        target: Tensor,
        noise_std: Tensor,
        prior_mean: Tensor,
        prior_cov: Tensor,
        emission_coeff: Tensor,
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
            Log probability, shape (batch_size, )
        Tensor
            Filtered_mean, shape (batch_size, latent_dim)
        Tensor
            Filtered_covariance, shape (batch_size, latent_dim, latent_dim)
        """
        # H * mu (batch_size, 1)
        target_mean = (emission_coeff.permute(0, 2, 1) @ prior_mean.unsqueeze(-1)).squeeze(-1)
        # v (batch_size, 1)
        residual = target - target_mean
        # R (batch_size, 1, 1)
        noise_cov = torch.diag_embed(noise_std * noise_std)
        # F (batch_size, 1, 1)
        target_cov = emission_coeff.permute(0, 2, 1) @ prior_cov @ emission_coeff + noise_cov
        # K (batch_size, latent_dim, 1)
        kalman_gain = prior_cov @ emission_coeff @ torch.inverse(target_cov)

        # mu = mu_t + K * v (batch_size, latent_dim)
        filtered_mean = (prior_mean.unsqueeze(-1) + kalman_gain @ residual).squeeze(-1)
        # P = (I - KH)P_t (batch_size, latent_dim, latent_dim)
        filtered_cov = (torch.eye(self.latent_dim) - kalman_gain @ emission_coeff.permute(0, 2, 1)) @ prior_cov
        # log-likelihood (batch_size, )
        log_p = torch.distributions.multivariate_normal.MultivariateNormal(target_mean, target_cov).log_prob(target)
        return log_p, filtered_mean, filtered_cov

    def kalman_filter(self, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs Kalman filtering given observations.

        Parameters
        ----------
        targets
            Observations, shape (batch_size, seq_length, output_dim)

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
        targets = targets.permute(1, 0, 2)
        log_p_seq = []
        mean = self.prior_mean
        cov = self.prior_cov

        for t in range(self.seq_length):
            log_p, filtered_mean, filtered_cov = self.kalman_filter_step(
                target=targets[t],
                noise_std=self.noise_std[t],
                prior_mean=mean,
                prior_cov=cov,
                emission_coeff=self.emission_coeff[t],
            )

            log_p_seq.append(log_p.unsqueeze(-1))
            mean = (self.transition_coeff @ filtered_mean.unsqueeze(-1)).squeeze(-1)
            cov = self.transition_coeff @ filtered_cov @ self.transition_coeff.permute(0, 2, 1) + self.innovation_coeff[
                t
            ] @ self.innovation_coeff[t].permute(0, 2, 1)

        return torch.cat(log_p_seq), mean, cov

    def log_likelihood(self, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the log-likelihood of the target.

        Parameters
        ----------
        targets:
            Tensor with targets of shape (batch_size, seq_length, output_dim)

        Returns
        -------
        Tensor
            Tensor with log-likelihoods of targets of shape (batch_size, seq_length)
        """

        log_p, final_mean, final_cov = self.kalman_filter(targets=targets)

        return log_p, final_mean, final_cov
