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
        prior_std: Tensor,  # (batch_size, latent_dim)
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
        prior_std:
            Prior standard deviation for latent state with shape (batch_size, latent_dim)
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
        self.prior_std = prior_std
        self.offset = offset
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        self.batch_size = self.prior_mean.shape[0]
        self.prior_cov: Tensor = torch.diag_embed(prior_std * prior_std)  # (batch_size, latent_dim, latent_dim)

    def _sample_initials(self, n_samples: int):
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
