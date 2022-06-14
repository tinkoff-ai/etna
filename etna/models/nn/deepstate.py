from typing import Tuple
from typing import TypedDict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor


class TrainBatch(TypedDict):
    encoder_real: Tensor
    target: Tensor


class InferenceBatch(TypedDict):
    encoder_real: Tensor


class DeepStateNetwork(LightningModule):
    def __init__(self, input_size: int, seq_length: int,  latent_dim: int, num_layers: int = 1):
        super(DeepStateNetwork).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.seq_length = seq_length

        self.RNN = nn.LSTM(
            num_layers=self.num_layers, hidden_size=self.latent_dim, input_size=self.input_size, batch_first=False
        )
        self.projectors = nn.ModuleDict(
            dict(
                prior_mean=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                prior_cov=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                noise_std=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                innovation=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                residuals=nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
            )
        )

    def training_step(self, train_batch: TrainBatch, batch_idx):
        encoder_real = train_batch["encoder_real"].float().permute(1, 0, 2)  # (encoder_length, batch_size, input_size)
        targets = train_batch["target"].float().permute(1, 0)  # (encoder_length, batch_size)

        output, (_, _) = self.RNN(encoder_real)  # (batch_size, encoder_length, latent_dim)
        lds = LDS(
            emission_coeff=self.projectors["emission_coeff"](output),
            transition_coeff=self.projectors["transition_coeff"](output),
            innovation_coeff=self.projectors["innovation_coeff"](output),
            noise_std=self.projectors["noise_std"](output),
            residuals=self.projectors["residuals"](output),
            prior_mean=self.projectors["prior_mean"](output[:, 0]),
            prior_cov=self.projectors["prior_cov"](output[:, 0]),
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
        residuals: Tensor,
        prior_mean: Tensor,
        prior_cov: Tensor,
        seq_length: int,
        latent_dim: int,
    ):
        self.emission_coeff = emission_coeff
        self.transition_coeff = transition_coeff
        self.innovation_coeff = innovation_coeff
        self.noise_std = noise_std
        self.residuals = residuals
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.seq_length = seq_length
        self.latent_dim = latent_dim

    def kalman_filter_step(
        self,
        target: Tensor,
        residual: Tensor,
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
        residual
            Residual component, shape (batch_size, output_dim)
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
        # H * mu (batch_size, output_dim)
        target_mean = torch.matmul(emission_coeff, prior_mean.unsqueeze(-1)).squeeze(-1)
        # R (batch_size, output_dim, output_dim)
        noise_cov = torch.diag_embed(noise_std * noise_std)
        # F (batch_size, output_dim, output_dim)
        target_cov = torch.matmul(torch.matmul(emission_coeff, prior_cov), emission_coeff.permute(0, 2, 1)) + noise_cov
        # K (batch_size, latent_dim, output_dim)
        kalman_gain = torch.matmul(torch.matmul(prior_cov, emission_coeff.permute(0, 2, 1)), torch.inverse(target_cov))

        # mu = mu_t + K * v (batch_size, latent_dim)
        filtered_mean = prior_mean + torch.matmul(kalman_gain, residual.unsqueeze(-1)).squeeze(-1)
        # P = (I - KH)P_t (batch_size, latent_dim, latent_dim)
        filtered_cov = torch.matmul(torch.eye(self.latent_dim) - torch.matmul(kalman_gain, emission_coeff), prior_cov)
        # log-likelihood (batch_size, )
        log_p = torch.distributions.multivariate_normal.MultivariateNormal(target_mean, target_cov).log_prob(target)

        return log_p, filtered_mean, filtered_cov

    def kalman_filter(self, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs Kalman filtering given observations.

        Parameters
        ----------
        targets
            Observations, shape (seq_length, batch_size, output_dim)

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
                target=targets[t],
                residual=self.residuals[t],
                noise_std=self.noise_std[t],
                prior_mean=mean,
                prior_cov=cov,
                emission_coeff=self.emission_coeff[t],
            )

            log_p_seq.append(log_p.unsqueeze(-1))
            mean = torch.matmul(self.transition_coeff[t], filtered_mean.unsqueeze(-1)).squeeze(-1)
            cov = torch.matmul(
                torch.matmul(self.transition_coeff[t], filtered_cov), self.transition_coeff[t].permute(0, 2, 1)
            ) + torch.matmul(self.innovation_coeff[t], self.innovation_coeff[t].permute(0, 2, 1))

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
