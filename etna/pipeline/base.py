import math
import warnings
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy.stats import norm
from typing_extensions import TypedDict
from typing_extensions import assert_never

from etna.core import AbstractSaveable
from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.loggers import tslogger
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode
from etna.metrics.functional_metrics import ArrayLike

Timestamp = Union[str, pd.Timestamp]


class CrossValidationMode(str, Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} modes allowed"
        )


class FoldMask(BaseMixin):
    """Container to hold the description of the fold mask.

    Fold masks are expected to be used for backtest strategy customization.
    """

    def __init__(
        self,
        first_train_timestamp: Optional[Timestamp],
        last_train_timestamp: Timestamp,
        target_timestamps: List[Timestamp],
    ):
        """Init FoldMask.

        Parameters
        ----------
        first_train_timestamp:
            First train timestamp, the first timestamp in the dataset if None is passed
        last_train_timestamp:
            Last train timestamp
        target_timestamps:
            List of target timestamps
        """
        self.first_train_timestamp = pd.to_datetime(first_train_timestamp) if first_train_timestamp else None
        self.last_train_timestamp = pd.to_datetime(last_train_timestamp)
        self.target_timestamps = sorted([pd.to_datetime(timestamp) for timestamp in target_timestamps])

        self._validate_last_train_timestamp()
        self._validate_target_timestamps()

    def _validate_last_train_timestamp(self):
        """Check that last train timestamp is later then first train timestamp."""
        if self.first_train_timestamp and self.last_train_timestamp < self.first_train_timestamp:
            raise ValueError("Last train timestamp should be not sooner than first train timestamp!")

    def _validate_target_timestamps(self):
        """Check that all target timestamps are later then last train timestamp."""
        first_target_timestamp = self.target_timestamps[0]
        if first_target_timestamp <= self.last_train_timestamp:
            raise ValueError("Target timestamps should be strictly later then last train timestamp!")

    def validate_on_dataset(self, ts: TSDataset, horizon: int):
        """Validate fold mask on the dataset with specified horizon.

        Parameters
        ----------
        ts:
            Dataset to validate on
        horizon:
            Forecasting horizon
        """
        dataset_timestamps = list(ts.index)
        dataset_description = ts.describe()

        min_first_timestamp = ts.index.min()
        if self.first_train_timestamp and self.first_train_timestamp < min_first_timestamp:
            raise ValueError(f"First train timestamp should be later than {min_first_timestamp}!")

        last_timestamp = dataset_description["end_timestamp"].min()
        if self.last_train_timestamp > last_timestamp:
            raise ValueError(f"Last train timestamp should be not later than {last_timestamp}!")

        dataset_first_target_timestamp = dataset_timestamps[dataset_timestamps.index(self.last_train_timestamp) + 1]
        mask_first_target_timestamp = self.target_timestamps[0]
        if mask_first_target_timestamp < dataset_first_target_timestamp:
            raise ValueError(f"First target timestamp should be not sooner than {dataset_first_target_timestamp}!")

        dataset_last_target_timestamp = dataset_timestamps[
            dataset_timestamps.index(self.last_train_timestamp) + horizon
        ]
        mask_last_target_timestamp = self.target_timestamps[-1]
        if dataset_last_target_timestamp < mask_last_target_timestamp:
            raise ValueError(f"Last target timestamp should be not later than {dataset_last_target_timestamp}!")


class AbstractPipeline(AbstractSaveable):
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
        :
            Fitted Pipeline instance
        """
        pass

    @abstractmethod
    def forecast(
        self,
        ts: Optional[TSDataset] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        n_folds: int = 3,
        return_components: bool = False,
    ) -> TSDataset:
        """Make a forecast of the next points of a dataset.

        The result of forecasting starts from the last point of ``ts``, not including it.

        Parameters
        ----------
        ts:
            Dataset to forecast. If not given, dataset given during :py:meth:``fit`` is used.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions
        """
        pass

    @abstractmethod
    def predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
    ) -> TSDataset:
        """Make in-sample predictions on dataset in a given range.

        Currently, in situation when segments start with different timestamps
        we only guarantee to work with ``start_timestamp`` >= beginning of all segments.

        Parameters
        ----------
        ts:
            Dataset to make predictions on.
        start_timestamp:
            First timestamp of prediction range to return, should be >= than first timestamp in ``ts``;
            expected that beginning of each segment <= ``start_timestamp``;
            if isn't set the first timestamp where each segment began is taken.
        end_timestamp:
            Last timestamp of prediction range to return; if isn't set the last timestamp of ``ts`` is taken.
            Expected that value is less or equal to the last timestamp in ``ts``.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions in ``[start_timestamp, end_timestamp]`` range.

        Raises
        ------
        ValueError:
            Value of ``end_timestamp`` is less than ``start_timestamp``.
        ValueError:
            Value of ``start_timestamp`` goes before point where each segment started.
        ValueError:
            Value of ``end_timestamp`` goes after the last timestamp.
        """

    @abstractmethod
    def backtest(
        self,
        ts: TSDataset,
        metrics: List[Metric],
        n_folds: Union[int, List[FoldMask]] = 5,
        mode: Optional[str] = None,
        aggregate_metrics: bool = False,
        n_jobs: int = 1,
        refit: Union[bool, int] = True,
        stride: Optional[int] = None,
        joblib_params: Optional[Dict[str, Any]] = None,
        forecast_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run backtest with the pipeline.

        If ``refit != True`` and some component of the pipeline doesn't support forecasting with gap, this component will raise an exception.

        Parameters
        ----------
        ts:
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds or the list of fold masks
        mode:
            Train generation policy: 'expand' or 'constant'. Works only if ``n_folds`` is integer.
            By default, is set to 'expand'.
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        refit:
            Determines how often pipeline should be retrained during iteration over folds.

            * If ``True``: pipeline is retrained on each fold.

            * If ``False``: pipeline is trained only on the first fold.

            * If ``value: int``: pipeline is trained every ``value`` folds starting from the first.

        stride:
            Number of points between folds. Works only if ``n_folds`` is integer. By default, is set to ``horizon``.
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`
        forecast_params:
            Additional parameters for :py:func:`~etna.pipeline.base.BasePipeline.forecast`

        Returns
        -------
        metrics_df, forecast_df, fold_info_df: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Metrics dataframe, forecast dataframe and dataframe with information about folds
        """

    @abstractmethod
    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get hyperparameter grid to tune.

        Returns
        -------
        :
            Grid with hyperparameters.
        """


class FoldParallelGroup(TypedDict):
    """Group for parallel fold processing."""

    train_fold_number: int
    train_mask: FoldMask
    forecast_fold_numbers: List[int]
    forecast_masks: List[FoldMask]


class _DummyMetric(Metric):
    """Dummy metric that is created only for implementation of BasePipeline._forecast_prediction_interval."""

    def __init__(self, mode: str = MetricAggregationMode.per_segment, **kwargs):
        super().__init__(mode=mode, metric_fn=self._compute_metric, **kwargs)

    @staticmethod
    def _compute_metric(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        return 0.0

    @property
    def greater_is_better(self) -> bool:
        return False

    def __call__(self, y_true: TSDataset, y_pred: TSDataset) -> Union[float, Dict[str, float]]:
        segments = set(y_true.df.columns.get_level_values("segment"))
        metrics_per_segment = {}
        for segment in segments:
            metrics_per_segment[segment] = 0.0
        metrics = self._aggregate_metrics(metrics_per_segment)
        return metrics


class BasePipeline(AbstractPipeline, BaseMixin):
    """Base class for pipeline."""

    def __init__(self, horizon: int):
        self._validate_horizon(horizon=horizon)
        self.horizon = horizon
        self.ts: Optional[TSDataset] = None

    @staticmethod
    def _validate_horizon(horizon: int):
        """Check that given number of folds is grater than 1."""
        if horizon <= 0:
            raise ValueError("At least one point in the future is expected.")

    @staticmethod
    def _validate_quantiles(quantiles: Sequence[float]) -> Sequence[float]:
        """Check that given number of folds is grater than 1."""
        for quantile in quantiles:
            if not (0 < quantile < 1):
                raise ValueError("Quantile should be a number from (0,1).")
        return quantiles

    @abstractmethod
    def _forecast(self, ts: TSDataset, return_components: bool) -> TSDataset:
        """Make predictions."""
        pass

    def _forecast_prediction_interval(
        self, ts: TSDataset, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Add prediction intervals to the forecasts."""
        with tslogger.disable():
            _, forecasts, _ = self.backtest(ts=ts, metrics=[_DummyMetric()], n_folds=n_folds)

        self._add_forecast_borders(ts=ts, backtest_forecasts=forecasts, quantiles=quantiles, predictions=predictions)

        return predictions

    @staticmethod
    def _validate_residuals_for_interval_estimation(backtest_forecasts: TSDataset, residuals: pd.DataFrame):
        len_backtest, num_segments = residuals.shape
        min_timestamp = backtest_forecasts.index.min()
        max_timestamp = backtest_forecasts.index.max()
        non_nan_counts = np.sum(~np.isnan(residuals.values), axis=0)
        if np.any(non_nan_counts < len_backtest):
            warnings.warn(
                f"There are NaNs in target on time span from {min_timestamp} to {max_timestamp}. "
                f"It can obstruct prediction interval estimation on history data."
            )
        if np.any(non_nan_counts < 2):
            raise ValueError(
                f"There aren't enough target values to evaluate prediction intervals on history! "
                f"For each segment there should be at least 2 points with defined value in a "
                f"time span from {min_timestamp} to {max_timestamp}. "
                f"You can try to increase n_folds parameter to make time span bigger."
            )

    def _add_forecast_borders(
        self, ts: TSDataset, backtest_forecasts: pd.DataFrame, quantiles: Sequence[float], predictions: TSDataset
    ) -> None:
        """Estimate prediction intervals and add to the forecasts."""
        backtest_forecasts = TSDataset(df=backtest_forecasts, freq=ts.freq)
        residuals = (
            backtest_forecasts.loc[:, pd.IndexSlice[:, "target"]]
            - ts[backtest_forecasts.index.min() : backtest_forecasts.index.max(), :, "target"]
        )

        self._validate_residuals_for_interval_estimation(backtest_forecasts=backtest_forecasts, residuals=residuals)
        sigma = np.nanstd(residuals.values, axis=0)

        borders = []
        for quantile in quantiles:
            z_q = norm.ppf(q=quantile)
            border = predictions[:, :, "target"] + sigma * z_q
            border.rename({"target": f"target_{quantile:.4g}"}, inplace=True, axis=1)
            borders.append(border)

        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

    def forecast(
        self,
        ts: Optional[TSDataset] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        n_folds: int = 3,
        return_components: bool = False,
    ) -> TSDataset:
        """Make a forecast of the next points of a dataset.

        The result of forecasting starts from the last point of ``ts``, not including it.

        Parameters
        ----------
        ts:
            Dataset to forecast. If not given, dataset given during :py:meth:``fit`` is used.
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions

        Raises
        ------
        NotImplementedError:
            Adding target components is not currently implemented
        """
        if ts is None:
            if self.ts is None:
                raise ValueError(
                    "There is no ts to forecast! Pass ts into forecast method or make sure that pipeline is loaded with ts."
                )
            ts = self.ts

        self._validate_quantiles(quantiles=quantiles)
        self._validate_backtest_n_folds(n_folds=n_folds)

        predictions = self._forecast(ts=ts, return_components=return_components)
        if prediction_interval:
            predictions = self._forecast_prediction_interval(
                ts=ts, predictions=predictions, quantiles=quantiles, n_folds=n_folds
            )
        return predictions

    @staticmethod
    def _make_predict_timestamps(
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        min_timestamp = ts.describe()["start_timestamp"].max()
        max_timestamp = ts.index[-1]

        if start_timestamp is None:
            start_timestamp = min_timestamp
        if end_timestamp is None:
            end_timestamp = max_timestamp

        if start_timestamp < min_timestamp:
            raise ValueError("Value of start_timestamp is less than beginning of some segments!")
        if end_timestamp > max_timestamp:
            raise ValueError("Value of end_timestamp is more than ending of dataset!")

        if start_timestamp > end_timestamp:
            raise ValueError("Value of end_timestamp is less than start_timestamp!")

        return start_timestamp, end_timestamp

    @abstractmethod
    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp],
        end_timestamp: Optional[pd.Timestamp],
        prediction_interval: bool,
        quantiles: Sequence[float],
        return_components: bool,
    ) -> TSDataset:
        pass

    def predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
        return_components: bool = False,
    ) -> TSDataset:
        """Make in-sample predictions on dataset in a given range.

        Currently, in situation when segments start with different timestamps
        we only guarantee to work with ``start_timestamp`` >= beginning of all segments.

        Parameters
        ----------
        ts:
            Dataset to make predictions on.
        start_timestamp:
            First timestamp of prediction range to return, should be >= than first timestamp in ``ts``;
            expected that beginning of each segment <= ``start_timestamp``;
            if isn't set the first timestamp where each segment began is taken.
        end_timestamp:
            Last timestamp of prediction range to return; if isn't set the last timestamp of ``ts`` is taken.
            Expected that value is less or equal to the last timestamp in ``ts``.
        prediction_interval:
            If True returns prediction interval for forecast.
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval.
        return_components:
            If True additionally returns forecast components

        Returns
        -------
        :
            Dataset with predictions in ``[start_timestamp, end_timestamp]`` range.

        Raises
        ------
        ValueError:
            Value of ``end_timestamp`` is less than ``start_timestamp``.
        ValueError:
            Value of ``start_timestamp`` goes before point where each segment started.
        ValueError:
            Value of ``end_timestamp`` goes after the last timestamp.
        NotImplementedError:
            Adding target components is not currently implemented
        """
        start_timestamp, end_timestamp = self._make_predict_timestamps(
            ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )
        self._validate_quantiles(quantiles=quantiles)
        result = self._predict(
            ts=ts,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            prediction_interval=prediction_interval,
            quantiles=quantiles,
            return_components=return_components,
        )
        return result

    def _init_backtest(self):
        self._folds: Optional[Dict[int, Any]] = None
        self._fold_column = "fold_number"

    @staticmethod
    def _validate_backtest_n_folds(n_folds: int):
        """Check that given n_folds value is >= 1."""
        if n_folds < 1:
            raise ValueError(f"Folds number should be a positive number, {n_folds} given")

    @staticmethod
    def _validate_backtest_mode(n_folds: Union[int, List[FoldMask]], mode: Optional[str]) -> CrossValidationMode:
        if mode is None:
            return CrossValidationMode.expand

        if not isinstance(n_folds, int):
            raise ValueError("Mode shouldn't be set if n_folds are fold masks!")

        return CrossValidationMode(mode.lower())

    @staticmethod
    def _validate_backtest_stride(n_folds: Union[int, List[FoldMask]], horizon: int, stride: Optional[int]) -> int:
        if stride is None:
            return horizon

        if not isinstance(n_folds, int):
            raise ValueError("Stride shouldn't be set if n_folds are fold masks!")

        if stride < 1:
            raise ValueError(f"Stride should be a positive number, {stride} given!")

        return stride

    @staticmethod
    def _validate_backtest_dataset(ts: TSDataset, n_folds: int, horizon: int, stride: int):
        """Check all segments have enough timestamps to validate forecaster with given number of splits."""
        min_required_length = horizon + (n_folds - 1) * stride
        segments = set(ts.df.columns.get_level_values("segment"))
        for segment in segments:
            segment_target = ts[:, segment, "target"]
            if len(segment_target) < min_required_length:
                raise ValueError(
                    f"All the series from feature dataframe should contain at least "
                    f"{horizon} + {n_folds-1} * {stride} = {min_required_length} timestamps; "
                    f"series {segment} does not."
                )

    @staticmethod
    def _generate_masks_from_n_folds(
        ts: TSDataset, n_folds: int, horizon: int, mode: CrossValidationMode, stride: int
    ) -> List[FoldMask]:
        """Generate fold masks from n_folds."""
        if mode is CrossValidationMode.expand:
            constant_history_length = 0
        elif mode is CrossValidationMode.constant:
            constant_history_length = 1
        else:
            assert_never(mode)

        masks = []
        dataset_timestamps = list(ts.index)
        min_timestamp_idx, max_timestamp_idx = 0, len(dataset_timestamps)
        for offset in range(n_folds, 0, -1):
            min_train_idx = min_timestamp_idx + (n_folds - offset) * stride * constant_history_length
            max_train_idx = max_timestamp_idx - stride * (offset - 1) - horizon - 1
            min_test_idx = max_train_idx + 1
            max_test_idx = max_train_idx + horizon

            min_train, max_train = dataset_timestamps[min_train_idx], dataset_timestamps[max_train_idx]
            min_test, max_test = dataset_timestamps[min_test_idx], dataset_timestamps[max_test_idx]

            mask = FoldMask(
                first_train_timestamp=min_train,
                last_train_timestamp=max_train,
                target_timestamps=list(pd.date_range(start=min_test, end=max_test, freq=ts.freq)),
            )
            masks.append(mask)

        return masks

    @staticmethod
    def _validate_backtest_metrics(metrics: List[Metric]):
        """Check that given metrics are valid for backtest."""
        if not metrics:
            raise ValueError("At least one metric required")
        for metric in metrics:
            if not metric.mode == MetricAggregationMode.per_segment:
                raise ValueError(
                    f"All the metrics should be in {MetricAggregationMode.per_segment}, "
                    f"{metric.name} metric is in {metric.mode} mode"
                )

    @staticmethod
    def _generate_folds_datasets(
        ts: TSDataset, masks: List[FoldMask], horizon: int
    ) -> Generator[Tuple[TSDataset, TSDataset], None, None]:
        """Generate folds."""
        timestamps = list(ts.index)
        for mask in masks:
            min_train_idx = timestamps.index(mask.first_train_timestamp)
            max_train_idx = timestamps.index(mask.last_train_timestamp)
            min_test_idx = max_train_idx + 1
            max_test_idx = max_train_idx + horizon

            min_train, max_train = timestamps[min_train_idx], timestamps[max_train_idx]
            min_test, max_test = timestamps[min_test_idx], timestamps[max_test_idx]

            train, test = ts.train_test_split(
                train_start=min_train, train_end=max_train, test_start=min_test, test_end=max_test
            )
            yield train, test

    def _compute_metrics(
        self, metrics: List[Metric], y_true: TSDataset, y_pred: TSDataset
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for given y_true, y_pred."""
        metrics_values: Dict[str, Dict[str, float]] = {}
        for metric in metrics:
            metrics_values[metric.name] = metric(y_true=y_true, y_pred=y_pred)  # type: ignore
        return metrics_values

    def _fit_backtest_pipeline(
        self,
        ts: TSDataset,
        fold_number: int,
    ) -> "BasePipeline":
        """Fit pipeline for a given data in backtest."""
        tslogger.start_experiment(job_type="training", group=str(fold_number))
        pipeline = deepcopy(self)
        pipeline.fit(ts=ts)
        tslogger.finish_experiment()
        return pipeline

    def _forecast_backtest_pipeline(
        self, pipeline: "BasePipeline", ts: TSDataset, fold_number: int, forecast_params: Dict[str, Any]
    ) -> TSDataset:
        """Make a forecast with a given pipeline in backtest."""
        tslogger.start_experiment(job_type="forecasting", group=str(fold_number))
        forecast = pipeline.forecast(ts=ts, **forecast_params)
        tslogger.finish_experiment()
        return forecast

    def _process_fold_forecast(
        self,
        forecast: TSDataset,
        train: TSDataset,
        test: TSDataset,
        pipeline: "BasePipeline",
        fold_number: int,
        mask: FoldMask,
        metrics: List[Metric],
    ) -> Dict[str, Any]:
        """Process forecast made for a fold."""
        tslogger.start_experiment(job_type="crossval", group=str(fold_number))

        fold: Dict[str, Any] = {}
        for stage_name, stage_df in zip(("train", "test"), (train, test)):
            fold[f"{stage_name}_timerange"] = {}
            fold[f"{stage_name}_timerange"]["start"] = stage_df.index.min()
            fold[f"{stage_name}_timerange"]["end"] = stage_df.index.max()

        forecast.df = forecast.df.loc[mask.target_timestamps]
        test.df = test.df.loc[mask.target_timestamps]

        fold["forecast"] = forecast
        fold["metrics"] = deepcopy(pipeline._compute_metrics(metrics=metrics, y_true=test, y_pred=forecast))

        tslogger.log_backtest_run(pd.DataFrame(fold["metrics"]), forecast.to_pandas(), test.to_pandas())
        tslogger.finish_experiment()

        return fold

    def _get_backtest_metrics(self, aggregate_metrics: bool = False) -> pd.DataFrame:
        """Get dataframe with metrics."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        metrics_dfs = []

        for i, fold in self._folds.items():
            fold_metrics = pd.DataFrame(fold["metrics"]).reset_index().rename({"index": "segment"}, axis=1)
            fold_metrics[self._fold_column] = i
            metrics_dfs.append(fold_metrics)
        metrics_df = pd.concat(metrics_dfs)
        metrics_df.sort_values(["segment", self._fold_column], inplace=True)

        if aggregate_metrics:
            metrics_df = metrics_df.groupby("segment").mean().reset_index().drop(self._fold_column, axis=1)

        return metrics_df

    def _get_fold_info(self) -> pd.DataFrame:
        """Get information about folds."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        timerange_dfs = []
        for fold_number, fold_info in self._folds.items():
            tmp_df = pd.DataFrame()
            for stage_name in ("train", "test"):
                for border in ("start", "end"):
                    tmp_df[f"{stage_name}_{border}_time"] = [fold_info[f"{stage_name}_timerange"][border]]
            tmp_df[self._fold_column] = fold_number
            timerange_dfs.append(tmp_df)
        timerange_df = pd.concat(timerange_dfs, ignore_index=True)
        return timerange_df

    def _get_backtest_forecasts(self) -> pd.DataFrame:
        """Get forecasts from different folds."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        forecasts_list = []
        for fold_number, fold_info in self._folds.items():
            forecast_ts = fold_info["forecast"]
            segments = forecast_ts.segments
            forecast = forecast_ts.df
            fold_number_df = pd.DataFrame(
                np.tile(fold_number, (forecast.index.shape[0], len(segments))),
                columns=pd.MultiIndex.from_product([segments, [self._fold_column]], names=("segment", "feature")),
                index=forecast.index,
            )
            forecast = forecast.join(fold_number_df)
            forecasts_list.append(forecast)
        forecasts = pd.concat(forecasts_list)
        forecasts.sort_index(axis=1, inplace=True)
        return forecasts

    def _prepare_fold_masks(
        self, ts: TSDataset, masks: Union[int, List[FoldMask]], mode: CrossValidationMode, stride: int
    ) -> List[FoldMask]:
        """Prepare and validate fold masks."""
        if isinstance(masks, int):
            self._validate_backtest_n_folds(n_folds=masks)
            self._validate_backtest_dataset(ts=ts, n_folds=masks, horizon=self.horizon, stride=stride)
            masks = self._generate_masks_from_n_folds(
                ts=ts, n_folds=masks, horizon=self.horizon, mode=mode, stride=stride
            )
        for i, mask in enumerate(masks):
            mask.first_train_timestamp = mask.first_train_timestamp if mask.first_train_timestamp else ts.index[0]
            masks[i] = mask
        for mask in masks:
            mask.validate_on_dataset(ts=ts, horizon=self.horizon)
        return masks

    @staticmethod
    def _make_backtest_fold_groups(masks: List[FoldMask], refit: Union[bool, int]) -> List[FoldParallelGroup]:
        """Make groups of folds for backtest."""
        if not refit:
            refit = len(masks)

        grouped_folds = []
        num_groups = math.ceil(len(masks) / refit)
        for group_id in range(num_groups):
            train_fold_number = group_id * refit
            forecast_fold_numbers = [train_fold_number + i for i in range(refit) if train_fold_number + i < len(masks)]
            cur_group: FoldParallelGroup = {
                "train_fold_number": train_fold_number,
                "train_mask": masks[train_fold_number],
                "forecast_fold_numbers": forecast_fold_numbers,
                "forecast_masks": [masks[i] for i in forecast_fold_numbers],
            }
            grouped_folds.append(cur_group)

        return grouped_folds

    def _run_all_folds(
        self,
        masks: List[FoldMask],
        ts: TSDataset,
        metrics: List[Metric],
        n_jobs: int,
        refit: Union[bool, int],
        joblib_params: Dict[str, Any],
        forecast_params: Dict[str, Any],
    ) -> Dict[int, Any]:
        """Run pipeline on all folds."""
        fold_groups = self._make_backtest_fold_groups(masks=masks, refit=refit)

        with Parallel(n_jobs=n_jobs, **joblib_params) as parallel:
            # fitting
            fit_masks = [group["train_mask"] for group in fold_groups]
            fit_datasets = (
                train for train, _ in self._generate_folds_datasets(ts=ts, masks=fit_masks, horizon=self.horizon)
            )
            pipelines = parallel(
                delayed(self._fit_backtest_pipeline)(ts=fit_ts, fold_number=fold_groups[group_idx]["train_fold_number"])
                for group_idx, fit_ts in enumerate(fit_datasets)
            )

            # forecasting
            forecast_masks = [group["forecast_masks"] for group in fold_groups]
            forecast_datasets = (
                (
                    train
                    for train, _ in self._generate_folds_datasets(
                        ts=ts, masks=group_forecast_masks, horizon=self.horizon
                    )
                )
                for group_forecast_masks in forecast_masks
            )
            forecasts_flat = parallel(
                delayed(self._forecast_backtest_pipeline)(
                    ts=forecast_ts,
                    pipeline=pipelines[group_idx],
                    fold_number=fold_groups[group_idx]["forecast_fold_numbers"][idx],
                    forecast_params=forecast_params,
                )
                for group_idx, group_forecast_datasets in enumerate(forecast_datasets)
                for idx, forecast_ts in enumerate(group_forecast_datasets)
            )

            # processing forecasts
            fold_process_train_datasets = (
                train for train, _ in self._generate_folds_datasets(ts=ts, masks=fit_masks, horizon=self.horizon)
            )
            fold_process_test_datasets = (
                (
                    test
                    for _, test in self._generate_folds_datasets(
                        ts=ts, masks=group_forecast_masks, horizon=self.horizon
                    )
                )
                for group_forecast_masks in forecast_masks
            )
            fold_results_flat = parallel(
                delayed(self._process_fold_forecast)(
                    forecast=forecasts_flat[group_idx * refit + idx],
                    train=train,
                    test=test,
                    pipeline=pipelines[group_idx],
                    fold_number=fold_groups[group_idx]["forecast_fold_numbers"][idx],
                    mask=fold_groups[group_idx]["forecast_masks"][idx],
                    metrics=metrics,
                )
                for group_idx, (train, group_fold_process_test_datasets) in enumerate(
                    zip(fold_process_train_datasets, fold_process_test_datasets)
                )
                for idx, test in enumerate(group_fold_process_test_datasets)
            )

        results = {
            fold_number: fold_results_flat[group_idx * refit + idx]
            for group_idx in range(len(fold_groups))
            for idx, fold_number in enumerate(fold_groups[group_idx]["forecast_fold_numbers"])
        }
        return results

    def backtest(
        self,
        ts: TSDataset,
        metrics: List[Metric],
        n_folds: Union[int, List[FoldMask]] = 5,
        mode: Optional[str] = None,
        aggregate_metrics: bool = False,
        n_jobs: int = 1,
        refit: Union[bool, int] = True,
        stride: Optional[int] = None,
        joblib_params: Optional[Dict[str, Any]] = None,
        forecast_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run backtest with the pipeline.

        If ``refit != True`` and some component of the pipeline doesn't support forecasting with gap, this component will raise an exception.

        Parameters
        ----------
        ts:
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds or the list of fold masks
        mode:
            Train generation policy: 'expand' or 'constant'. Works only if ``n_folds`` is integer.
            By default, is set to 'expand'.
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        refit:
            Determines how often pipeline should be retrained during iteration over folds.

            * If ``True``: pipeline is retrained on each fold.

            * If ``False``: pipeline is trained only on the first fold.

            * If ``value: int``: pipeline is trained every ``value`` folds starting from the first.

        stride:
            Number of points between folds. Works only if ``n_folds`` is integer. By default, is set to ``horizon``.
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`
        forecast_params:
            Additional parameters for :py:func:`~etna.pipeline.base.BasePipeline.forecast`

        Returns
        -------
        metrics_df, forecast_df, fold_info_df: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Metrics dataframe, forecast dataframe and dataframe with information about folds

        Raises
        ------
        ValueError:
            If ``mode`` is set when ``n_folds`` are ``List[FoldMask]``.
        ValueError:
            If ``stride`` is set when ``n_folds`` are ``List[FoldMask]``.
        """
        mode_enum = self._validate_backtest_mode(n_folds=n_folds, mode=mode)
        stride = self._validate_backtest_stride(n_folds=n_folds, horizon=self.horizon, stride=stride)

        if joblib_params is None:
            joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")

        if forecast_params is None:
            forecast_params = dict()

        self._init_backtest()
        self._validate_backtest_metrics(metrics=metrics)
        masks = self._prepare_fold_masks(ts=ts, masks=n_folds, mode=mode_enum, stride=stride)
        self._folds = self._run_all_folds(
            masks=masks,
            ts=ts,
            metrics=metrics,
            n_jobs=n_jobs,
            refit=refit,
            joblib_params=joblib_params,
            forecast_params=forecast_params,
        )

        metrics_df = self._get_backtest_metrics(aggregate_metrics=aggregate_metrics)
        forecast_df = self._get_backtest_forecasts()
        fold_info_df = self._get_fold_info()

        tslogger.start_experiment(job_type="crossval_results", group="all")
        tslogger.log_backtest_metrics(ts, metrics_df, forecast_df, fold_info_df)
        tslogger.finish_experiment()

        return metrics_df, forecast_df, fold_info_df
