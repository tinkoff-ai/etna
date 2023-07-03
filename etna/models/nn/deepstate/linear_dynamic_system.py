from typing import Tuple

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from etna.core import BaseMixin


class LDS(BaseMixin):
    """Class which implements Linear Dynamical System (LDS) as a distribution."""

    def __init__(
        self,
        emission_coeff: Tensor,  # (batch_size, seq_length, latent_dim)
        transition_coeff: Tensor,  # (latent_dim, latent_dim)
        innovation_coeff: Tensor,  # (batch_size, seq_length, latent_dim)
        noise_std: Tensor,  # (batch_size, seq_length, 1)
        prior_mean: Tensor,  # (batch_size, latent_dim)
        prior_cov: Tensor,  # (batch_size, latent_dim, latent_dim)
        offset: Tensor,  # (batch_size, seq_length, 1)
        seq_length: int,
        latent_dim: int,
    ):
        """Create instance of LDS.

        Parameters
        ----------
        emission_coeff:
            Emission coefficient matrix with shape (batch_size, seq_length, latent_dim).
        transition_coeff:
            Transition coefficient matrix with shape (latent_dim, latent_dim).
        innovation_coeff:
            Innovation coefficient matrix with shape (batch_size, seq_length, latent_dim).
        noise_std:
            Noise standard deviation for targets with shape (batch_size, seq_length, 1).
        prior_mean:
            Prior mean for latent state with shape (batch_size, latent_dim)
        prior_cov:
            Prior covariance matrix for latent state with shape (batch_size, latent_dim, latent_dim)
        offset:
            Offset for the target with shape (batch_size, seq_length, 1)
        seq_length:
            Length of the sequence.
        latent_dim:
            Dimension of the latent space.
        """
        self.emission_coeff = emission_coeff
        self.transition_coeff = transition_coeff
        self.innovation_coeff = innovation_coeff
        self.noise_std = noise_std
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.offset = offset
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        self.batch_size = self.prior_mean.shape[0]
        self._eye = torch.eye(self.latent_dim).to(noise_std)

    def kalman_filter_step(
        self,
        target: Tensor,  # (batch_size, 1)
        noise_std: Tensor,  # (batch_size, 1)
        prior_mean: Tensor,  # (batch_size, latent_dim)
        prior_cov: Tensor,  # (batch_size, latent_dim, latent_dim)
        emission_coeff: Tensor,  # (batch_size, latent_dim)
        offset: Tensor,  # (batch_size, 1)
    ):
        """One step of the Kalman filter.

        This function computes the filtered state (mean and covariance) given the
        LDS coefficients in the prior state (mean and variance) and observations.

        Parameters
        ----------
        target:
            Observations of the system with shape (batch_size, 1)
        noise_std:
            Standard deviation of the observations noise with shape (batch_size, 1)
        prior_mean:
            Prior mean of the latent state with shape (batch_size, latent_dim)
        prior_cov:
            Prior covariance of the latent state with shape (batch_size, latent_dim, latent_dim)
        emission_coeff:
            Emission coefficient with shape (batch_size, latent_dim)
        offset:
            Offset for the observations with shape (batch_size, 1)

        Returns
        -------
        :
            Log probability with shape (batch_size, 1)
        :
            Filtered_mean with shape (batch_size, latent_dim, 1)
        :
            Filtered_covariance with shape (batch_size, latent_dim, latent_dim)
        """
        emission_coeff = emission_coeff.unsqueeze(-1)

        # H * mu (batch_size, 1)
        target_mean = (emission_coeff.permute(0, 2, 1) @ prior_mean.unsqueeze(-1)).squeeze(-1)
        # v (batch_size, 1)
        residual = target - target_mean - offset
        # R (batch_size, 1, 1)
        noise_cov = torch.diag_embed(noise_std * noise_std)
        # F (batch_size, 1, 1)
        target_cov = emission_coeff.permute(0, 2, 1) @ prior_cov @ emission_coeff + noise_cov
        # K (batch_size, latent_dim)
        kalman_gain = (prior_cov @ emission_coeff @ torch.inverse(target_cov)).squeeze(-1)

        # mu = mu_t + K * v (batch_size, latent_dim)
        filtered_mean = prior_mean + (kalman_gain.unsqueeze(-1) @ residual.unsqueeze(-1)).squeeze(-1)
        # P = (I - KH)P_t (batch_size, latent_dim, latent_dim)
        filtered_cov = (self._eye.to(target) - kalman_gain.unsqueeze(-1) @ emission_coeff.permute(0, 2, 1)) @ prior_cov
        # log-likelihood (batch_size, 1)
        log_p = (
            Normal(target_mean.squeeze(-1), torch.sqrt(target_cov.squeeze(-1).squeeze(-1)))
            .log_prob(target.squeeze(-1))
            .unsqueeze(-1)
        )
        return log_p, filtered_mean, filtered_cov

    def kalman_filter(self, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform Kalman filtering of given observations.

        Parameters
        ----------
        targets:
            Tensor with observations with shape (batch_size, seq_length, 1)

        Returns
        -------
        :
            Log probabilities with shape shape (batch_size, seq_length)
        :
            Mean of p(l_T | l_{T-1}), where T is seq_length, with shape (batch_size, latent_dim)
        :
            Covariance of p(l_T | l_{T-1}), where T is seq_length, with shape (batch_size, latent_dim, latent_dim)
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

        log_p = torch.cat(log_p_seq, dim=1)
        return log_p, mean, cov

    def log_likelihood(self, targets: Tensor) -> Tensor:
        """Compute the log-likelihood of the target.

        Parameters
        ----------
        targets:
            Tensor with targets of shape (batch_size, seq_length, 1)

        Returns
        -------
        :
            Tensor with log-likelihoods of target of shape (batch_size, seq_length)
        """
        log_p, _, _ = self.kalman_filter(targets=targets)

        return log_p

    def _sample_initials(self, n_samples: int):
        """Sample initial values for noise and latent state."""
        # (n_samples, batch_size, seq_length, latent_dim)
        eps_latent = MultivariateNormal(
            loc=torch.zeros(self.latent_dim), covariance_matrix=torch.eye(self.latent_dim)
        ).sample((n_samples, self.batch_size, self.seq_length))

        # (n_samples, batch_size, seq_length, 1)
        eps_observation = Normal(loc=0, scale=1).sample((n_samples, self.batch_size, self.seq_length, 1))

        # (n_samples, batch_size, latent_dim)
        l_0 = MultivariateNormal(loc=self.prior_mean, covariance_matrix=self.prior_cov).sample((n_samples,))
        return l_0, eps_latent, eps_observation

    def sample(self, n_samples: int) -> Tensor:
        """Sample the trajectories of targets from the current LDS.

        Parameters
        ----------
        n_samples:
            Number of trajectories to sample.

        Returns
        -------
        :
            Tensor with trajectories with shape (n_samples, batch_size, seq_length, 1).
        """
        l_t, eps_latent, eps_observation = self._sample_initials(n_samples=n_samples)
        samples_seq = []
        for t in range(self.seq_length):
            a_t = self.emission_coeff[:, t].unsqueeze(-1).permute(0, 2, 1) @ l_t.unsqueeze(-1)
            b_t = self.offset[:, t].unsqueeze(0).unsqueeze(-1)
            noise_t = (self.noise_std[:, t].unsqueeze(0) * eps_observation[:, :, t]).unsqueeze(-1)
            z_t = a_t + b_t + noise_t
            samples_seq.append(z_t)

            a_t = (self.transition_coeff @ l_t.unsqueeze(-1)).squeeze(-1)
            noise_t = self.innovation_coeff[:, t].unsqueeze(0) * eps_latent[:, :, t]
            l_t = a_t + noise_t

        # (n_samples, batch_size, seq_length, 1)
        samples = torch.cat(samples_seq, dim=2)
        return samples
