from typing import Optional

import numpy as np

from etna.transforms.decomposition.change_points_based.per_interval_models.base import PerIntervalModel


class ConstantPerIntervalModel(PerIntervalModel):
    """ConstantPerIntervalModel gives a constant prediction it was fitted with."""

    def __init__(self):
        """Init ConstantPerIntervalModel."""
        self.value: Optional[float] = None

    def fit(self, value: float) -> "ConstantPerIntervalModel":
        """Fit constant model.

        Parameters
        ----------
        value:
            constant value to be used for prediction

        Returns
        -------
        self:
            fitted ConstantPerIntervalModel
        """
        self.value = value
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict with constant.

        Parameters
        ----------
        features:
            features to make prediction for

        Returns
        -------
        prediction:
            constant array of features' len
        """
        prediction = np.ones(shape=(features.shape[0],)) * self.value
        return prediction
