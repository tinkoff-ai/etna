import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression

from etna.transforms.decomposition.change_points_based.per_interval_models.base import PerIntervalModel


class SklearnPerIntervalModel(PerIntervalModel):
    """SklearnPerIntervalModel applies PerIntervalModel interface for sklearn-like regression models."""

    def __init__(self, model: RegressorMixin = LinearRegression()):
        """Init SklearnPerIntervalModel.

        Parameters
        ----------
        model:
            model with sklearn interface to use for interval processing
        """
        self.model = model

    def fit(self, features: np.ndarray, target: np.ndarray) -> "SklearnPerIntervalModel":
        """Fit model with given features and targets.

        Parameters
        ----------
        features:
            features to fit model with
        target:
            targets to fit model

        Returns
        -------
        self:
            fitted SklearnPerIntervalModel
        """
        self.model.fit(X=features, y=target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction for given features.

        Parameters
        ----------
        features:
            features to make prediction for

        Returns
        -------
        prediction:
            model's prediction for given features
        """
        return self.model.predict(X=features)


class SklearnPreprocessing2PerIntervalModel(PerIntervalModel):
    """SklearnPreprocessing2PerIntervalModel applies PerIntervalModel interface for sklearn preprocessings."""

    def __init__(self, preprocessing: TransformerMixin):
        self.preprocessing = preprocessing

    def fit(self, features: np.ndarray, target: np.ndarray) -> "SklearnPreprocessing2PerIntervalModel":
        """Fit preprocessing with given features and targets.

        Parameters
        ----------
        features:
            features to fit preprocessing with
        target:
            targets to apply preprocessing to

        Returns
        -------
        self:
            fitted SklearnPreprocessing2PerIntervalModel
        """
        self.preprocessing.fit(X=features, y=target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing to given features.

        Parameters
        ----------
        features:
            features to make preprocessing for

        Returns
        -------
        prediction:
            preprocessing's prediction for given features
        """
        prediction = self.preprocessing.transform(X=features)
        return prediction
