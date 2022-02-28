import warnings
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import pandas as pd

from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.metrics import Metric


class AbstractPipeline(ABC):
    """Interface for pipeline."""

    @abstractmethod
    def fit(self, ts: TSDataset) -> "AbstractPipeline":
        """Fit the Pipeline.

        Parameters
        ----------
        ts:
            Dataset with timeseries data

        Returns
        -------
        self:
            Fitted Pipeline instance
        """
        pass

    @abstractmethod
    def forecast(self, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975)) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval

        Returns
        -------
        forecast:
            Dataset with predictions
        """
        pass

    @abstractmethod
    def backtest(
        self,
        ts: TSDataset,
        metrics: List[Metric],
        n_folds: int = 5,
        mode: str = "expand",
        aggregate_metrics: bool = False,
        n_jobs: int = 1,
        joblib_params: Dict[str, Any] = dict(verbose=11, backend="multiprocessing", mmap_mode="c"),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run backtest with the pipeline.

        Parameters
        ----------
        ts:
            dataset to fit models in backtest
        metrics:
            list of metrics to compute for each fold
        n_folds:
            number of folds
        mode:
            one of 'expand', 'constant' -- train generation policy
        aggregate_metrics:
            if True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            number of jobs to run in parallel
        joblib_params:
            additional parameters for joblib.Parallel

        Returns
        -------
        metrics_df, forecast_df, fold_info_df:
            metrics dataframe, forecast dataframe and dataframe with information about folds
        """


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
