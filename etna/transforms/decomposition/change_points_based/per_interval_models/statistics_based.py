from typing import Callable
from typing import Optional

import numpy as np

from etna.transforms.decomposition.change_points_based.per_interval_models.base import PerIntervalModel


class StatisticsPerIntervalModel(PerIntervalModel):
    """StatisticsPerIntervalModel gets statistics from series and use them for prediction."""

    def __init__(self, statistics_function: Callable[[np.ndarray], float]):
        """Init StatisticsPerIntervalModel.

        Parameters
        ----------
        statistics_function:
            function to compute statistics from series
        """
        self.statistics_function = statistics_function
        self._statistics_value: Optional[float] = None

    def fit(self, features: np.ndarray, target: np.ndarray, *args, **kwargs) -> "StatisticsPerIntervalModel":
        """Fit statistics from given target.

        Parameters
        ----------
        features:
            features of the series, will be ignored
        target:
            target to compute statistics for

        Returns
        -------
        self:
            fitted StatisticsPerIntervalModel
        """
        self._statistics_value = self.statistics_function(target)
        return self

    def predict(self, features: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Build prediction from precomputed statistics.

        Parameters
        ----------
        features:
            features to build prediction for

        Returns
        -------
        prediction:
            array of features len filled with statistics value
        """
        prediction = np.full(shape=(features.shape[0],), fill_value=self._statistics_value)
        return prediction


class MeanPerIntervalModel(StatisticsPerIntervalModel):
    """MeanPerIntervalModel.

    MeanPerIntervalModel is a shortcut for
    :py:class:`etna.transforms.decomposition.change_points_based.per_interval_models.statistics_based.StatisticsPerIntervalModel
    that uses mean value as statistics function.
    """

    def __init__(self):
        super().__init__(statistics_function=np.mean)


class MedianPerIntervalModel(StatisticsPerIntervalModel):
    """MedianPerIntervalModel.

    MedianPerIntervalModel is a shortcut for
    :py:class:`etna.transforms.decomposition.change_points_based.per_interval_models.statistics_based.StatisticsPerIntervalModel
    that uses median value as statistics function.
    """

    def __init__(self):
        super().__init__(statistics_function=np.median)
