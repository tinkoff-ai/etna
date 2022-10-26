import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

from etna.transforms.decomposition.changepoints_based.per_interval_models.base import PerIntervalModel


class SklearnPerIntervalModel(PerIntervalModel):
    def __init__(self, model: RegressorMixin = LinearRegression()):
        self.model = model

    def fit(self, features: np.ndarray, target: np.ndarray) -> "SklearnPerIntervalModel":
        self.model.fit(X=features, y=target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(X=features)


class SklearnPreprocessing2Model(PerIntervalModel):
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing

    def fit(self, features: np.ndarray, target: np.ndarray) -> "SklearnPreprocessing2Model":
        self.preprocessing.fit(X=features, y=target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        prediction = self.preprocessing.transform(X=features)
        return prediction
