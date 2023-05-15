from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna import SETTINGS
from etna.analysis import absolute_difference_distance
from etna.analysis import get_anomalies_density
from etna.analysis import get_anomalies_median
from etna.analysis import get_anomalies_prediction_interval
from etna.datasets import TSDataset
from etna.models import SARIMAXModel
from etna.transforms.outliers.base import OutliersTransform

if SETTINGS.prophet_required:
    from etna.models import ProphetModel

if SETTINGS.auto_required:
    from optuna.distributions import BaseDistribution
    from optuna.distributions import CategoricalDistribution
    from optuna.distributions import IntUniformDistribution
    from optuna.distributions import UniformDistribution


class MedianOutliersTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.median_outliers.get_anomalies_median` to find anomalies in data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(self, in_column: str, window_size: int = 10, alpha: float = 3):
        """Create instance of MedianOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        window_size:
            number of points in the window
        alpha:
            coefficient for determining the threshold
        """
        self.window_size = window_size
        self.alpha = alpha
        super().__init__(in_column=in_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call :py:func:`~etna.analysis.outliers.median_outliers.get_anomalies_median` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_median(ts=ts, in_column=self.in_column, window_size=self.window_size, alpha=self.alpha)

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``window_size``, ``alpha``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "window_size": IntUniformDistribution(low=3, high=30),
            "alpha": UniformDistribution(low=0.5, high=5),
        }


class DensityOutliersTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density` to find anomalies in data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        window_size: int = 15,
        distance_coef: float = 3,
        n_neighbors: int = 3,
        distance_func: Callable[[float, float], float] = absolute_difference_distance,
    ):
        """Create instance of DensityOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        window_size:
            size of windows to build
        distance_coef:
            factor for standard deviation that forms distance threshold to determine points are close to each other
        n_neighbors:
            min number of close neighbors of point not to be outlier
        distance_func:
            distance function
        """
        self.window_size = window_size
        self.distance_coef = distance_coef
        self.n_neighbors = n_neighbors
        self.distance_func = distance_func
        super().__init__(in_column=in_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_density(
            ts=ts,
            in_column=self.in_column,
            window_size=self.window_size,
            distance_coef=self.distance_coef,
            n_neighbors=self.n_neighbors,
            distance_func=self.distance_func,
        )

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``window_size``, ``distance_coef``, ``n_neighbors``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "window_size": IntUniformDistribution(low=3, high=30),
            "distance_coef": UniformDistribution(low=0.5, high=5),
            "n_neighbors": IntUniformDistribution(low=1, high=10),
        }


class PredictionIntervalOutliersTransform(OutliersTransform):
    """Transform that uses :py:func:`~etna.analysis.outliers.prediction_interval_outliers.get_anomalies_prediction_interval` to find anomalies in data."""

    def __init__(
        self,
        in_column: str,
        model: Union[Literal["prophet"], Literal["sarimax"], Type["ProphetModel"], Type["SARIMAXModel"]],
        interval_width: float = 0.95,
        **model_kwargs,
    ):
        """Create instance of PredictionIntervalOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        model:
            model for prediction interval estimation
        interval_width:
            width of the prediction interval

        Notes
        -----
        For not "target" column only column data will be used for learning.
        """
        self.model = model
        self.interval_width = interval_width
        self.model_kwargs = model_kwargs
        self._model_type = self._get_model_type(model)
        super().__init__(in_column=in_column)

    @staticmethod
    def _get_model_type(
        model: Union[Literal["prophet"], Literal["sarimax"], Type["ProphetModel"], Type["SARIMAXModel"]]
    ) -> Union[Type["ProphetModel"], Type["SARIMAXModel"]]:
        if isinstance(model, str):
            if model == "prophet":
                return ProphetModel
            elif model == "sarimax":
                return SARIMAXModel
        return model

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call :py:func:`~etna.analysis.outliers.prediction_interval_outliers.get_anomalies_prediction_interval` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        :
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_anomalies_prediction_interval(
            ts=ts,
            model=self._model_type,
            interval_width=self.interval_width,
            in_column=self.in_column,
            **self.model_kwargs,
        )

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``interval_width``, ``model``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "interval_width": UniformDistribution(low=0.8, high=1.0),
            "model": CategoricalDistribution(["prophet", "sarimax"]),
        }


__all__ = [
    "MedianOutliersTransform",
    "DensityOutliersTransform",
    "PredictionIntervalOutliersTransform",
]
