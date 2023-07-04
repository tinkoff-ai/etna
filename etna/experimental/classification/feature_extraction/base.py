from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

import numpy as np

from etna.core import BaseMixin
from etna.experimental.classification.base import PickleSerializable


class BaseTimeSeriesFeatureExtractor(ABC, BaseMixin, PickleSerializable):
    """Base class for time series feature extractor."""

    @abstractmethod
    def fit(self, x: List[np.ndarray], y: Optional[np.ndarray] = None) -> "BaseTimeSeriesFeatureExtractor":
        """Fit the feature extractor.

        Parameters
        ----------
        x:
            Array with time series.
        y:
            Array of class labels.

        Returns
        -------
        :
            Fitted instance of feature extractor.
        """
        pass

    @abstractmethod
    def transform(self, x: List[np.ndarray]) -> np.ndarray:
        """Extract features from the input data.

        Parameters
        ----------
        x:
            Array with time series.

        Returns
        -------
        :
            Transformed input data.
        """
        pass

    def fit_transform(self, x: List[np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the feature extractor and extract features from the input data.

        Parameters
        ----------
        x:
            Array with time series.
        y:
            Array of class labels.

        Returns
        -------
         :
            Transformed input data.
        """
        return self.fit(x=x, y=y).transform(x=x)
