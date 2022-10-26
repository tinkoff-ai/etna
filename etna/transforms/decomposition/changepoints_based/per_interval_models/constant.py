from typing import Optional

import numpy as np

from etna.transforms.decomposition.changepoints_based.per_interval_models.base import PerIntervalModel


class ConstantPerIntervalModel(PerIntervalModel):
    def __init__(self):
        self.value: Optional[float] = None

    def fit(self, value: float) -> "ConstantPerIntervalModel":
        self.value = value
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        prediction = np.ones(shape=(features.shape[0],)) * self.value
        return prediction
