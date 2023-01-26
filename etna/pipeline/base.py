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

from etna.core import AbstractSaveable
from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode

Timestamp = Union[str, pd.Timestamp]


class CrossValidationMode(Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"


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
        self, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975), n_folds: int = 3
    ) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation

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
        mode: str = "expand",
        aggregate_metrics: bool = False,
        n_jobs: int = 1,
        joblib_params: Optional[Dict[str, Any]] = None,
        forecast_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run backtest with the pipeline.

        Parameters
        ----------
        ts:
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds or the list of fold masks
        mode:
            One of 'expand', 'constant' -- train generation policy
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`
        forecast_params:
            Additional parameters for :py:func:`~etna.pipeline.base.BasePipeline.forecast`

        Returns
        -------
        metrics_df, forecast_df, fold_info_df: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Metrics dataframe, forecast dataframe and dataframe with information about folds
        """


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
    def _forecast(self) -> TSDataset:
        """Make predictions."""
        pass

    def _forecast_prediction_interval(
        self, predictions: TSDataset, quantiles: Sequence[float], n_folds: int
    ) -> TSDataset:
        """Add prediction intervals to the forecasts."""
        if self.ts is None:
            raise ValueError("Pipeline is not fitted! Fit the Pipeline before calling forecast method.")
        with tslogger.disable():
            _, forecasts, _ = self.backtest(ts=self.ts, metrics=[MAE()], n_folds=n_folds)

        self._add_forecast_borders(backtest_forecasts=forecasts, quantiles=quantiles, predictions=predictions)

        return predictions

    def _add_forecast_borders(
        self, backtest_forecasts: pd.DataFrame, quantiles: Sequence[float], predictions: TSDataset
    ) -> None:
        """Estimate prediction intervals and add to the forecasts."""
        if self.ts is None:
            raise ValueError("Pipeline is not fitted!")

        backtest_forecasts = TSDataset(df=backtest_forecasts, freq=self.ts.freq)
        residuals = (
            backtest_forecasts.loc[:, pd.IndexSlice[:, "target"]]
            - self.ts[backtest_forecasts.index.min() : backtest_forecasts.index.max(), :, "target"]
        )

        sigma = np.std(residuals.values, axis=0)
        borders = []
        for quantile in quantiles:
            z_q = norm.ppf(q=quantile)
            border = predictions[:, :, "target"] + sigma * z_q
            border.rename({"target": f"target_{quantile:.4g}"}, inplace=True, axis=1)
            borders.append(border)

        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

    def forecast(
        self, prediction_interval: bool = False, quantiles: Sequence[float] = (0.025, 0.975), n_folds: int = 3
    ) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        prediction_interval:
            If True returns prediction interval for forecast
        quantiles:
            Levels of prediction distribution. By default 2.5% and 97.5% taken to form a 95% prediction interval
        n_folds:
            Number of folds to use in the backtest for prediction interval estimation

        Returns
        -------
        :
            Dataset with predictions
        """
        if self.ts is None:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted! Fit the {self.__class__.__name__} "
                f"before calling forecast method."
            )
        self._validate_quantiles(quantiles=quantiles)
        self._validate_backtest_n_folds(n_folds=n_folds)

        predictions = self._forecast()
        if prediction_interval:
            predictions = self._forecast_prediction_interval(
                predictions=predictions, quantiles=quantiles, n_folds=n_folds
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

    def _predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp],
        end_timestamp: Optional[pd.Timestamp],
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> TSDataset:
        raise NotImplementedError("Predict method isn't implemented!")

    def predict(
        self,
        ts: TSDataset,
        start_timestamp: Optional[pd.Timestamp] = None,
        end_timestamp: Optional[pd.Timestamp] = None,
        prediction_interval: bool = False,
        quantiles: Sequence[float] = (0.025, 0.975),
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
        )
        return result

    def _init_backtest(self):
        self._folds: Optional[Dict[int, Any]] = None
        self._fold_column = "fold_number"

    @staticmethod
    def _validate_backtest_n_folds(n_folds: int):
        """Check that given n_folds value is valid."""
        if n_folds < 1:
            raise ValueError(f"Folds number should be a positive number, {n_folds} given")

    @staticmethod
    def _validate_backtest_dataset(ts: TSDataset, n_folds: int, horizon: int):
        """Check all segments have enough timestamps to validate forecaster with given number of splits."""
        min_required_length = horizon * n_folds
        segments = set(ts.df.columns.get_level_values("segment"))
        for segment in segments:
            segment_target = ts[:, segment, "target"]
            if len(segment_target) < min_required_length:
                raise ValueError(
                    f"All the series from feature dataframe should contain at least "
                    f"{horizon} * {n_folds} = {min_required_length} timestamps; "
                    f"series {segment} does not."
                )

    @staticmethod
    def _generate_masks_from_n_folds(ts: TSDataset, n_folds: int, horizon: int, mode: str) -> List[FoldMask]:
        """Generate fold masks from n_folds."""
        mode_enum = CrossValidationMode(mode.lower())
        if mode_enum == CrossValidationMode.expand:
            constant_history_length = 0
        elif mode_enum == CrossValidationMode.constant:
            constant_history_length = 1
        else:
            raise NotImplementedError(
                f"Only '{CrossValidationMode.expand}' and '{CrossValidationMode.constant}' modes allowed"
            )

        masks = []
        dataset_timestamps = list(ts.index)
        min_timestamp_idx, max_timestamp_idx = 0, len(dataset_timestamps)
        for offset in range(n_folds, 0, -1):
            min_train_idx = min_timestamp_idx + (n_folds - offset) * horizon * constant_history_length
            max_train_idx = max_timestamp_idx - horizon * offset - 1
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

    def _run_fold(
        self,
        train: TSDataset,
        test: TSDataset,
        fold_number: int,
        mask: FoldMask,
        metrics: List[Metric],
        forecast_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run fit-forecast pipeline of model for one fold."""
        tslogger.start_experiment(job_type="crossval", group=str(fold_number))

        pipeline = deepcopy(self)
        pipeline.fit(ts=train)
        forecast = pipeline.forecast(**forecast_params)
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
        timerange_df = pd.concat(timerange_dfs)
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

    def _prepare_fold_masks(self, ts: TSDataset, masks: Union[int, List[FoldMask]], mode: str) -> List[FoldMask]:
        """Prepare and validate fold masks."""
        if isinstance(masks, int):
            self._validate_backtest_n_folds(n_folds=masks)
            self._validate_backtest_dataset(ts=ts, n_folds=masks, horizon=self.horizon)
            masks = self._generate_masks_from_n_folds(ts=ts, n_folds=masks, horizon=self.horizon, mode=mode)
        for i, mask in enumerate(masks):
            mask.first_train_timestamp = mask.first_train_timestamp if mask.first_train_timestamp else ts.index[0]
            masks[i] = mask
        for mask in masks:
            mask.validate_on_dataset(ts=ts, horizon=self.horizon)
        return masks

    def backtest(
        self,
        ts: TSDataset,
        metrics: List[Metric],
        n_folds: Union[int, List[FoldMask]] = 5,
        mode: str = "expand",
        aggregate_metrics: bool = False,
        n_jobs: int = 1,
        joblib_params: Optional[Dict[str, Any]] = None,
        forecast_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run backtest with the pipeline.

        Parameters
        ----------
        ts:
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds or the list of fold masks
        mode:
            One of 'expand', 'constant' -- train generation policy, ignored if n_folds is a list of masks
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for :py:class:`joblib.Parallel`
        forecast_params:
            Additional parameters for :py:func:`~etna.pipeline.base.BasePipeline.forecast`

        Returns
        -------
        metrics_df, forecast_df, fold_info_df: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Metrics dataframe, forecast dataframe and dataframe with information about folds
        """
        if joblib_params is None:
            joblib_params = dict(verbose=11, backend="multiprocessing", mmap_mode="c")

        if forecast_params is None:
            forecast_params = dict()

        self._init_backtest()
        self._validate_backtest_metrics(metrics=metrics)
        masks = self._prepare_fold_masks(ts=ts, masks=n_folds, mode=mode)

        folds = Parallel(n_jobs=n_jobs, **joblib_params)(
            delayed(self._run_fold)(
                train=train,
                test=test,
                fold_number=fold_number,
                mask=masks[fold_number],
                metrics=metrics,
                forecast_params=forecast_params,
            )
            for fold_number, (train, test) in enumerate(
                self._generate_folds_datasets(ts=ts, masks=masks, horizon=self.horizon)
            )
        )
        self._folds = {i: fold for i, fold in enumerate(folds)}

        metrics_df = self._get_backtest_metrics(aggregate_metrics=aggregate_metrics)
        forecast_df = self._get_backtest_forecasts()
        fold_info_df = self._get_fold_info()

        tslogger.start_experiment(job_type="crossval_results", group="all")
        tslogger.log_backtest_metrics(ts, metrics_df, forecast_df, fold_info_df)
        tslogger.finish_experiment()

        return metrics_df, forecast_df, fold_info_df
