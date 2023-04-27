from abc import ABC
from abc import abstractmethod

import numpy as np

from etna.core import BaseMixin


class PerIntervalModel(BaseMixin, ABC):
    """Class to handle intervals in change point based transforms.

    PerIntervalModel is a class to process intervals between change points
    in :py:mod:`~etna.transforms.decomposition.change_points_based` transforms.
    """

    @abstractmethod
    def fit(self, features: np.ndarray, target: np.ndarray, *args, **kwargs) -> "PerIntervalModel":
        """Fit per interval model with given params."""
        pass

    @abstractmethod
    def predict(self, features: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Make prediction with per interval model."""
        pass
