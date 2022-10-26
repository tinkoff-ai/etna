from abc import ABC
from abc import abstractmethod

import numpy as np


class PerIntervalModel(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs) -> "PerIntervalModel":
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        pass
