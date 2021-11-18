import warnings
from abc import ABC
from abc import abstractmethod
from typing import Sequence

from etna.core import BaseMixin


class BasePipeline(ABC, BaseMixin):
    """Base class for all pipelines."""

    def __init__(
        self,
        quantiles: Sequence[float],
    ):
        """
        Create instance of Pipeline with given parameters.

        Parameters
        ----------
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval

        Raises
        ------
        ValueError:
            If the quantile is not within (0, 1).
        """
        self.quantiles = self._validate_quantiles(quantiles)

    @property
    @abstractmethod
    def support_prediction_interval(self) -> bool:
        """Indicate if method supports prediction intervals."""
        pass

    @staticmethod
    def _validate_quantiles(quantiles: Sequence[float]) -> Sequence[float]:
        """Check that given number of folds is grater than 1."""
        for quantile in quantiles:
            if not (0 < quantile < 1):
                raise ValueError("Quantile should be a number from (0,1).")
        return quantiles

    def check_support_prediction_interval(self, prediction_interval_option: bool = False):
        """Check if pipeline supports prediction intervals, if not, warns a user.

        Parameters
        ----------
        prediction_interval_option:
            indicate if forecast method is called with `prediction_interval=True`
        """
        if not self.support_prediction_interval and prediction_interval_option:
            warnings.warn("This class doesn't support prediction intervals and they won't be build")
