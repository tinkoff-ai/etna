from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Protocol
from typing_extensions import assert_never

from etna.core import BaseMixin
from etna.datasets.tsdataset import TSDataset
from etna.loggers import tslogger
from etna.metrics.functional_metrics import ArrayLike


class MetricAggregationMode(str, Enum):
    """Enum for different metric aggregation modes."""

    macro = "macro"
    per_segment = "per-segment"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregation allowed"
        )


class MetricFunctionSignature(str, Enum):
    """Enum for different metric function signatures."""

    #: function should expect arrays of y_pred and y_true with length ``n_timestamps`` and return scalar
    array_to_scalar = "array_to_scalar"

    #: function should expect matrices of y_pred and y_true with shape ``(n_timestamps, n_segments)``
    #: and return vector of length ``n_segments``
    matrix_to_array = "matrix_to_array"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} signatures allowed"
        )


class MetricFunction(Protocol):
    """Protocol for ``metric_fn`` parameter."""

    @abstractmethod
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
        pass


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

    def __init__(
        self,
        metric_fn: MetricFunction,
        mode: str = MetricAggregationMode.per_segment,
        metric_fn_signature: str = "array_to_scalar",
        **kwargs,
    ):
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

        metric_fn_signature:
            type of signature of ``metric_fn`` (see :py:class:`~etna.metrics.base.MetricFunctionSignature`)
        kwargs:
            functional metric's params

        Raises
        ------
        NotImplementedError:
            If non-existent ``mode`` is used.
        NotImplementedError:
            If non-existent ``metric_fn_signature`` is used.
        """
        if MetricAggregationMode(mode) is MetricAggregationMode.macro:
            self._aggregate_metrics = self._macro_average
        elif MetricAggregationMode(mode) is MetricAggregationMode.per_segment:
            self._aggregate_metrics = self._per_segment_average

        self._metric_fn_signature = MetricFunctionSignature(metric_fn_signature)

        self.metric_fn = metric_fn
        self.kwargs = kwargs
        self.mode = mode
        self.metric_fn_signature = metric_fn_signature

    @property
    def name(self) -> str:
        """Name of the metric for representation."""
        return self.__class__.__name__

    @staticmethod
    def _validate_segments(y_true: TSDataset, y_pred: TSDataset):
        """Check that segments in ``y_true`` and ``y_pred`` are the same.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            if there are mismatches in y_true and y_pred segments
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

    @staticmethod
    def _validate_target_columns(y_true: TSDataset, y_pred: TSDataset):
        """Check that all the segments from ``y_true`` and ``y_pred`` has 'target' column.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            if one of segments in y_true or y_pred doesn't contain 'target' column.
        """
        segments = set(y_true.df.columns.get_level_values("segment"))

        for segment in segments:
            for name, dataset in zip(("y_true", "y_pred"), (y_true, y_pred)):
                if (segment, "target") not in dataset.columns:
                    raise ValueError(
                        f"All the segments in {name} should contain 'target' column. Segment {segment} doesn't."
                    )

    @staticmethod
    def _validate_index(y_true: TSDataset, y_pred: TSDataset):
        """Check that ``y_true`` and ``y_pred`` have the same timestamps.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            If there are mismatches in ``y_true`` and ``y_pred`` timestamps
        """
        if not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred have different timestamps")

    @staticmethod
    def _validate_nans(y_true: TSDataset, y_pred: TSDataset):
        """Check that ``y_true`` and ``y_pred`` doesn't have NaNs.

        Parameters
        ----------
        y_true:
            y_true dataset
        y_pred:
            y_pred dataset

        Raises
        ------
        ValueError:
            If there are NaNs in ``y_true`` or ``y_pred``
        """
        df_true = y_true.df.loc[:, pd.IndexSlice[:, "target"]]
        df_pred = y_pred.df.loc[:, pd.IndexSlice[:, "target"]]

        df_true_isna = df_true.isna().any().any()
        if df_true_isna > 0:
            raise ValueError("There are NaNs in y_true")

        df_pred_isna = df_pred.isna().any().any()
        if df_pred_isna > 0:
            raise ValueError("There are NaNs in y_pred")

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
        self._validate_segments(y_true=y_true, y_pred=y_pred)
        self._validate_target_columns(y_true=y_true, y_pred=y_pred)
        self._validate_index(y_true=y_true, y_pred=y_pred)
        self._validate_nans(y_true=y_true, y_pred=y_pred)

        df_true = y_true[:, :, "target"].sort_index(axis=1)
        df_pred = y_pred[:, :, "target"].sort_index(axis=1)

        segments = df_true.columns.get_level_values("segment").unique()

        metrics_per_segment: Dict[str, float]
        if self._metric_fn_signature is MetricFunctionSignature.array_to_scalar:
            metrics_per_segment = {}
            for i, segment in enumerate(segments):
                cur_y_true = df_true.iloc[:, i].values
                cur_y_pred = df_pred.iloc[:, i].values
                metrics_per_segment[segment] = self.metric_fn(y_true=cur_y_true, y_pred=cur_y_pred, **self.kwargs)  # type: ignore
        elif self._metric_fn_signature is MetricFunctionSignature.matrix_to_array:
            values = self.metric_fn(y_true=df_true.values, y_pred=df_pred.values, **self.kwargs)
            metrics_per_segment = dict(zip(segments, values))  # type: ignore
        else:
            assert_never(self._metric_fn_signature)

        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics


__all__ = ["Metric", "MetricAggregationMode"]
