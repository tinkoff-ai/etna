from typing import Callable
from typing import Optional

import numpy as np

from etna.transforms.decomposition.changepoints_based.per_interval_models.base import PerIntervalModel


class StatisticsPerIntervalModel(PerIntervalModel):
    def __init__(self, statistics_function: Callable[[np.ndarray], np.ndarray]):
        self.statistics_function = statistics_function
        self._statistics_value: Optional[float] = 0

    def fit(self, features: np.ndarray, target: np.ndarray) -> "StatisticsPerIntervalModel":
        self._statistics_value = self.statistics_function(target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        prediction = np.ones(shape=(features.shape[0],)) * self._statistics_value
        return prediction


class MeanPerIntervalModel(StatisticsPerIntervalModel):
    def __init__(self):
        super().__init__(statistics_function=np.mean)


class MedianPerIntervalModel(StatisticsPerIntervalModel):
    def __init__(self):
        super().__init__(statistics_function=np.median)
