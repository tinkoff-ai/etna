from abc import ABC
from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from etna.core import BaseMixin


class SSM(ABC, BaseMixin):
    """Base class for State Space Model.

    The system dynamics is described with the following equations:
    .. math::
       y_t = a^T_tl_{t-1} + b_t + \sigma_t\epsilon_t
       l_t = F_tl_{t-1} + g_t\epsilon_t
       l_0 \sim N(\mu_0, diag(\sigma_0^2))
       y - state of the system
       l - state of the system in the latent space
       a - emission coefficient
       F - transition coefficient
       g - innovation coefficient
       \sigma - noise standard deviation
       \mu_0 - prior mean
       \sigma_0 - prior standard deviation
    """

    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        raise NotImplementedError

    @abstractmethod
    def emission_coeff(self, datetime_index: Tensor) -> Tensor:  # (batch_size, seq_length, latent_dim)
        """Emission coefficient matrix.

        Parameters
        ----------
        datetime_index:
            Tensor with the index values.
            Values should be from 0 to seasonal period.

        Returns
        -------
        :
            Emission coefficient matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_coeff(self, datetime_index: Tensor) -> Tensor:  # (latent_dim, latent_dim)
        """Transition coefficient matrix.

        Parameters
        ----------
        datetime_index:
            Tensor with the index values.
            Values should be from 0 to seasonal period.

        Returns
        -------
        :
            Transition coefficient matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:  # (batch_size, seq_length, latent_dim)
        """Innovation coefficient matrix.

        Parameters
        ----------
        datetime_index:
            Tensor with the index values.
            Values should be from 0 to seasonal period.

        Returns
        -------
        :
            Innovation coefficient matrix.
        """
        raise NotImplementedError


class LevelSSM(SSM):
    """Class for Level State Space Model."""

    def latent_dim(self) -> int:
        return 1

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
        batch_size, seq_length = datetime_index.shape[0], datetime_index.shape[1]
        emission_coeff = torch.ones(batch_size, seq_length, self.latent_dim())
        return emission_coeff.float()

    def transition_coeff(self, datetime_index: Tensor) -> Tensor:
        transition_coeff = torch.eye(self.latent_dim())
        return transition_coeff.float()

    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:
        return self.emission_coeff(datetime_index)


class LevelTrendSSM(LevelSSM):
    """Class for Level-Trend State Space Model."""

    def latent_dim(self) -> int:
        return 2

    def transition_coeff(self, datetime_index: Tensor) -> Tensor:
        transition_coeff = torch.eye(self.latent_dim())
        transition_coeff[0, 1] = 1
        return transition_coeff.float()


class SeasonalitySSM(LevelSSM):
    """Class for Seasonality State Space Model."""

    def __init__(self, num_seasons: int):
        """Create instance of _SingleDifferencingTransform.

        Parameters
        ----------
        num_seasons:
            Number of seasons in the considered seasonality period.
        """
        self.num_seasons = num_seasons

    def latent_dim(self) -> int:
        return self.num_seasons

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
        emission_coeff = one_hot(datetime_index.squeeze(-1), num_classes=self.latent_dim())
        return emission_coeff.float()
