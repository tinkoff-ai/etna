from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from etna.core import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger


class MetricAggregationMode(str, Enum):
    """Enum for different metric aggregation modes."""

    macro = "macro"
    per_segment = "per-segment"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregation allowed"
        )


class AbstractMetric(ABC):
    """Abstract class for metric."""

    @abstractmethod
    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
        """
        Compute metric's value with ``y_true`` and ``y_pred``.

        Notes
        -----
        Note that if ``y_true`` and ``y_pred`` are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
        :
            metric's value aggregated over segments or not (depends on mode)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass

    @property
    @abstractmethod
    def greater_is_better(self) -> Optional[bool]:
        """Whether higher metric value is better."""
        pass


class Metric(AbstractMetric, BaseMixin):
    """
    Base class for all the multi-segment metrics.

    How it works: Metric computes ``metric_fn`` value for each segment in given forecast
    dataset and aggregates it according to mode.
    """

    def __init__(self, metric_fn: Callable[..., float], mode: str = MetricAggregationMode.per_segment, **kwargs):
        """
        Init Metric.

        Parameters
        ----------
        metric_fn:
            functional metric
        mode:
            "macro" or "per-segment", way to aggregate metric values over segments:

            * if "macro" computes average value

            * if "per-segment" -- does not aggregate metrics

        kwargs:
            functional metric's params

        Raises
        ------
        NotImplementedError:
            it non existent mode is used
        """
        self.metric_fn = metric_fn
        self.kwargs = kwargs
        if MetricAggregationMode(mode) == MetricAggregationMode.macro:
            self._aggregate_metrics = self._macro_average
        elif MetricAggregationMode(mode) == MetricAggregationMode.per_segment:
            self._aggregate_metrics = self._per_segment_average
        self.mode = mode

    @property
    def name(self) -> str:
        """Name of the metric for representation."""
        return self.__class__.__name__

    @staticmethod
    def _validate_segment_columns(y_true: TSDataset, y_pred: TSDataset):
        """
        Check if all the segments from ``y_true`` are in ``y_pred`` and vice versa.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            if there are mismatches in y_true and y_pred segments,
        ValueError:
            if one of segments in y_true or y_pred doesn't contain 'target' column.
        """
        segments_true = set(y_true.df.columns.get_level_values("segment"))
        segments_pred = set(y_pred.df.columns.get_level_values("segment"))

        pred_diff_true = segments_pred - segments_true
        true_diff_pred = segments_true - segments_pred
        if pred_diff_true:
            raise ValueError(
                f"There are segments in y_pred that are not in y_true, for example: "
                f"{', '.join(list(pred_diff_true)[:5])}"
            )
        if true_diff_pred:
            raise ValueError(
                f"There are segments in y_true that are not in y_pred, for example: "
                f"{', '.join(list(true_diff_pred)[:5])}"
            )
        for segment in segments_true:
            for name, dataset in zip(("y_true", "y_pred"), (y_true, y_pred)):
                if "target" not in dataset.loc[:, segment].columns:
                    raise ValueError(
                        f"All the segments in {name} should contain 'target' column. Segment {segment} doesn't."
                    )

    @staticmethod
    def _validate_timestamp_columns(timestamp_true: pd.Series, timestamp_pred: pd.Series):
        """
        Check that ``y_true`` and ``y_pred`` have the same timestamp.

        Parameters
        ----------
        timestamp_true:
            y_true's timestamp column
        timestamp_pred:
            y_pred's timestamp column

        Raises
        ------
        ValueError:
            If there are mismatches in ``y_true`` and ``y_pred`` timestamps
        """
        if set(timestamp_pred) != set(timestamp_true):
            raise ValueError("y_true and y_pred have different timestamps")

    @staticmethod
    def _macro_average(metrics_per_segments: Dict[str, float]) -> Union[float, Dict[str, float]]:
        """
        Compute macro averaging of metrics over segment.

        Parameters
        ----------
        metrics_per_segments: dict of {segment: metric_value} for segments to aggregate

        Returns
        -------
        aggregated value of metric
        """
        return np.mean(list(metrics_per_segments.values())).item()

    @staticmethod
    def _per_segment_average(metrics_per_segments: Dict[str, float]) -> Union[float, Dict[str, float]]:
        """
        Compute per-segment averaging of metrics over segment.

        Parameters
        ----------
        metrics_per_segments: dict of {segment: metric_value} for segments to aggregate

        Returns
        -------
        aggregated dict of metric
        """
        return metrics_per_segments

    def _log_start(self):
        """Log metric computation."""
        tslogger.log(f"Metric {self.__repr__()} is calculated on dataset")

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
        """
        Compute metric's value with ``y_true`` and ``y_pred``.

        Notes
        -----
        Note that if ``y_true`` and ``y_pred`` are not sorted Metric will sort it anyway

        Parameters
        ----------
        y_true:
            dataset with true time series values
        y_pred:
            dataset with predicted time series values

        Returns
        -------
        :
            metric's value aggregated over segments or not (depends on mode)
        """
        self._log_start()
        self._validate_segment_columns(y_true=y_true, y_pred=y_pred)

        segments = set(y_true.df.columns.get_level_values("segment"))
        metrics_per_segment = {}
        for segment in segments:
            self._validate_timestamp_columns(
                timestamp_true=y_true[:, segment, "target"].dropna().index,
                timestamp_pred=y_pred[:, segment, "target"].dropna().index,
            )
            metrics_per_segment[segment] = self.metric_fn(
                y_true=y_true[:, segment, "target"].values, y_pred=y_pred[:, segment, "target"].values, **self.kwargs
            )
        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics


__all__ = ["Metric", "MetricAggregationMode"]
