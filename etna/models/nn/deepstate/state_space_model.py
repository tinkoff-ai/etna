from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from etna.core import BaseMixin


class SSM(ABC, BaseMixin):
    """Base class for State Space Model.

    The system dynamics is described with the following equations:

    .. math::
       y_t = a^T_t l_{t-1} + b_t + \sigma_t\\varepsilon_t
    .. math::
       l_t = F_t l_{t-1} + g_t\epsilon_t
    .. math::
       l_0 \sim N(\mu_0, diag(\sigma_0^2)), \\varepsilon_t \sim N(0, 1), \epsilon_t \sim N(0, 1),

    where

       :math:`y` - state of the system

       :math:`l` - state of the system in the latent space

       :math:`a` - emission coefficient

       :math:`F` - transition coefficient

       :math:`g` - innovation coefficient

       :math:`\sigma` - noise standard deviation

       :math:`\mu_0` - prior mean

       :math:`\sigma_0` - prior standard deviation
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

    @abstractmethod
    def generate_datetime_index(self, timestamps: np.ndarray) -> np.ndarray:
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        timestamps:
            Array with timestamps.

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        raise NotImplementedError


class LevelSSM(SSM):
    """Class for Level State Space Model."""

    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        return 1

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
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
        batch_size, seq_length = datetime_index.shape[0], datetime_index.shape[1]
        emission_coeff = torch.ones(batch_size, seq_length, self.latent_dim(), device=datetime_index.device)
        return emission_coeff.float()

    def transition_coeff(self, datetime_index: Tensor) -> Tensor:
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
        transition_coeff = torch.eye(self.latent_dim(), device=datetime_index.device)
        return transition_coeff.float()

    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:
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
        return self.emission_coeff(datetime_index)

    def generate_datetime_index(self, timestamps: np.ndarray) -> np.ndarray:
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        timestamps:
            Array with timestamps.

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        seq_length = timestamps.shape[0]
        return np.zeros(shape=(seq_length,))


class LevelTrendSSM(LevelSSM):
    """Class for Level-Trend State Space Model."""

    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        return 2

    def transition_coeff(self, datetime_index: Tensor) -> Tensor:
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
        transition_coeff = torch.eye(self.latent_dim(), device=datetime_index.device)
        transition_coeff[0, 1] = 1
        return transition_coeff.float()


class SeasonalitySSM(LevelSSM):
    """Class for Seasonality State Space Model."""

    def __init__(self, num_seasons: int, timestamp_transform: Callable[[pd.Timestamp], int]):
        """Create instance of SeasonalitySSM.

        Parameters
        ----------
        num_seasons:
            Number of seasons in the considered seasonality period.
        """
        self.num_seasons = num_seasons
        self.timestamp_transform = timestamp_transform

    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        return self.num_seasons

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
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
        emission_coeff = one_hot(datetime_index.squeeze(-1), num_classes=self.latent_dim())
        return emission_coeff.float()

    def generate_datetime_index(self, timestamps: np.ndarray) -> np.ndarray:
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        timestamps:
            Array with timestamps.

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        return np.array([self.timestamp_transform(timestamp) for timestamp in timestamps])


class YearlySeasonalitySSM(SeasonalitySSM):
    """Class for Yearly Seasonality State Space Model."""

    def __init__(self):
        super().__init__(num_seasons=12, timestamp_transform=self.get_timestamp_transform)

    def get_timestamp_transform(self, x: pd.Timestamp):
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        x:
            timestamp

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        return x.month - 1


class WeeklySeasonalitySSM(SeasonalitySSM):
    """Class for Weekly Seasonality State Space Model."""

    def __init__(self):
        super().__init__(num_seasons=7, timestamp_transform=self.get_timestamp_transform)

    def get_timestamp_transform(self, x: pd.Timestamp):
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        x:
            timestamp

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        return x.weekday()


class DaylySeasonalitySSM(SeasonalitySSM):
    """Class for Daily Seasonality State Space Model."""

    def __init__(self):
        super().__init__(num_seasons=24, timestamp_transform=self.get_timestamp_transform)

    def get_timestamp_transform(self, x: pd.Timestamp):
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        x:
            timestamp

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        return x.hour


class CompositeSSM(SSM):
    """Class to compose several State Space Models."""

    def __init__(
        self, seasonal_ssms: List[SeasonalitySSM], nonseasonal_ssm: Optional[Union[LevelSSM, LevelTrendSSM]] = None
    ):
        """Create instance of CompositeSSM.

        Parameters
        ----------
        seasonal_ssms:
            List with the instances of Seasonality State Space Models.
        nonseasonal_ssm:
            Instance of Level or Level-Trend State Space Model.
        """
        self.seasonal_ssms = seasonal_ssms
        self.nonseasonal_ssm = nonseasonal_ssm
        self.ssms: List[SSM] = self.seasonal_ssms  # type: ignore
        if self.nonseasonal_ssm is not None:
            self.ssms.append(self.nonseasonal_ssm)

    def latent_dim(self) -> int:
        """Dimension of the latent space.

        Returns
        -------
        :
            Dimension of the latent space.
        """
        return sum([ssm.latent_dim() for ssm in self.ssms])

    def emission_coeff(self, datetime_index: Tensor) -> Tensor:
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
        emission_coeff = torch.cat([ssm.emission_coeff(datetime_index[i]) for i, ssm in enumerate(self.ssms)], dim=-1)
        return emission_coeff

    def transition_coeff(self, datetime_index: Tensor) -> Tensor:
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
        place_holder = datetime_index[0]
        transition_coeff = torch.block_diag(*[ssm.transition_coeff(place_holder) for ssm in self.ssms])
        return transition_coeff

    def innovation_coeff(self, datetime_index: Tensor) -> Tensor:
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
        return self.emission_coeff(datetime_index)

    def generate_datetime_index(self, timestamps: np.ndarray) -> np.ndarray:
        """Generate datetime index to use in the State Space Model.

        Parameters
        ----------
        timestamps:
            Array with timestamps.

        Returns
        -------
        :
            Datetime index for State Space Model.
        """
        return np.vstack([ssm.generate_datetime_index(timestamps) for ssm in self.ssms])
