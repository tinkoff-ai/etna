import warnings
from abc import ABC
from abc import abstractmethod

from etna.core import BaseMixin


class BasePipeline(ABC, BaseMixin):
    """Base class for all pipelines."""

    def __init__(
        self,
        interval_width: float,
    ):
        """
        Create instance of Pipeline with given parameters.

        Parameters
        ----------
        interval_width:
            The significance level for the prediction interval.

        Raises
        ------
        ValueError:
            If the interval_width is not within (0, 1).
        """
        self.interval_width = self._validate_interval_width(interval_width)

    @property
    @abstractmethod
    def support_prediction_interval(self) -> bool:
        """Indicate if method supports prediction intervals."""
        pass

    @staticmethod
    def _validate_interval_width(interval_width: float) -> float:
        """Check that given number of folds is grater than 1."""
        if 0 < interval_width < 1:
            return interval_width
        else:
            raise ValueError("Interval width should be a number from (0,1).")

    def check_support_prediction_interval(self, prediction_interval_option: bool = False):
        """Check if pipeline supports prediction intervals, if not, warns a user.

        Parameters
        ----------
        prediction_interval_option:
            indicate if forecast method is called with `prediction_interval=True`
        """
        if not self.support_prediction_interval and prediction_interval_option:
            warnings.warn("This class doesn't support prediction intervals and they won't be build")
