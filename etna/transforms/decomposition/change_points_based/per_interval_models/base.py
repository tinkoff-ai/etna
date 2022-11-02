from abc import ABC
from abc import abstractmethod

import numpy as np


class PerIntervalModel(ABC):
    """PerIntervalModel is a class to process intervals between change points
    in `~etna.transforms.decomposition.change_points_based` transforms.
    """

    @abstractmethod
    def fit(self, *args, **kwargs) -> "PerIntervalModel":
        """Fit per interval model with given params."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """Make prediction with per interval model."""
        pass
