from abc import ABC
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

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel
from joblib import delayed
from scipy.stats import norm

from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import Metric
from etna.metrics import MetricAggregationMode


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
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds
        mode:
            One of 'expand', 'constant' -- train generation policy
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for joblib.Parallel

        Returns
        -------
        metrics_df, forecast_df, fold_info_df:
            Metrics dataframe, forecast dataframe and dataframe with information about folds
        """


class CrossValidationMode(Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"


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
        _, forecasts, _ = self.backtest(ts=self.ts, metrics=[MAE()], n_folds=n_folds)
        forecasts = TSDataset(df=forecasts, freq=self.ts.freq)
        residuals = (
            forecasts.loc[:, pd.IndexSlice[:, "target"]]
            - self.ts[forecasts.index.min() : forecasts.index.max(), :, "target"]
        )

        se = scipy.stats.sem(residuals)
        borders = []
        for quantile in quantiles:
            z_q = norm.ppf(q=quantile)
            border = predictions[:, :, "target"] + se * z_q
            border.rename({"target": f"target_{quantile:.4g}"}, inplace=True, axis=1)
            borders.append(border)

        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

        return predictions

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
        forecast:
            Dataset with predictions
        """
        if self.ts is None:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted! Fit the {self.__class__.__name__} before calling forecast method."
            )
        self._validate_quantiles(quantiles=quantiles)
        self._validate_backtest_n_folds(n_folds=n_folds)

        predictions = self._forecast()
        if prediction_interval:
            predictions = self._forecast_prediction_interval(
                predictions=predictions, quantiles=quantiles, n_folds=n_folds
            )
        return predictions

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
        ts: TSDataset, n_folds: int, horizon: int, mode: str = "expand"
    ) -> Generator[Tuple[TSDataset, TSDataset], None, None]:
        """Generate a sequence of train-test pairs according to timestamp."""
        mode_enum = CrossValidationMode[mode.lower()]
        if mode_enum == CrossValidationMode.expand:
            constant_history_length = 0
        elif mode_enum == CrossValidationMode.constant:
            constant_history_length = 1
        else:
            raise NotImplementedError(
                f"Only '{CrossValidationMode.expand}' and '{CrossValidationMode.constant}' modes allowed"
            )

        timestamps = ts.index
        min_timestamp_idx, max_timestamp_idx = 0, len(timestamps)
        for offset in range(n_folds, 0, -1):
            # if not self._constant_history_length, left border of train df is always equal to minimal timestamp value;
            # it means that all the given data is used.
            # if self._constant_history_length, left border of train df moves to one horizon steps on each split
            min_train_idx = min_timestamp_idx + (n_folds - offset) * horizon * constant_history_length
            max_train_idx = max_timestamp_idx - horizon * offset - 1
            min_test_idx = max_train_idx + 1
            max_test_idx = max_train_idx + horizon

            min_train, max_train = timestamps[min_train_idx], timestamps[max_train_idx]
            min_test, max_test = timestamps[min_test_idx], timestamps[max_test_idx]

            train, test = ts.train_test_split(
                train_start=min_train, train_end=max_train, test_start=min_test, test_end=max_test
            )

            yield train, test

    @staticmethod
    def _compute_metrics(metrics: List[Metric], y_true: TSDataset, y_pred: TSDataset) -> Dict[str, Dict[str, float]]:
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
        metrics: List[Metric],
    ) -> Dict[str, Any]:
        """Run fit-forecast pipeline of model for one fold."""
        tslogger.start_experiment(job_type="crossval", group=str(fold_number))

        pipeline = deepcopy(self)
        pipeline.fit(ts=train)
        forecast = pipeline.forecast()

        fold: Dict[str, Any] = {}
        for stage_name, stage_df in zip(("train", "test"), (train, test)):
            fold[f"{stage_name}_timerange"] = {}
            fold[f"{stage_name}_timerange"]["start"] = stage_df.index.min()
            fold[f"{stage_name}_timerange"]["end"] = stage_df.index.max()
        fold["forecast"] = forecast
        fold["metrics"] = deepcopy(self._compute_metrics(metrics=metrics, y_true=test, y_pred=forecast))

        tslogger.log_backtest_run(pd.DataFrame(fold["metrics"]), forecast.to_pandas(), test.to_pandas())
        tslogger.finish_experiment()

        return fold

    def _get_backtest_metrics(self, aggregate_metrics: bool = False) -> pd.DataFrame:
        """Get dataframe with metrics."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        metrics_df = pd.DataFrame()

        for i, fold in self._folds.items():
            fold_metrics = pd.DataFrame(fold["metrics"]).reset_index().rename({"index": "segment"}, axis=1)
            fold_metrics[self._fold_column] = i
            metrics_df = metrics_df.append(fold_metrics)

        metrics_df.sort_values(["segment", self._fold_column], inplace=True)

        if aggregate_metrics:
            metrics_df = metrics_df.groupby("segment").mean().reset_index().drop(self._fold_column, axis=1)

        return metrics_df

    def _get_fold_info(self) -> pd.DataFrame:
        """Get information about folds."""
        if self._folds is None:
            raise ValueError("Something went wrong during backtest initialization!")
        timerange_df = pd.DataFrame()
        for fold_number, fold_info in self._folds.items():
            tmp_df = pd.DataFrame()
            for stage_name in ("train", "test"):
                for border in ("start", "end"):
                    tmp_df[f"{stage_name}_{border}_time"] = [fold_info[f"{stage_name}_timerange"][border]]
            tmp_df[self._fold_column] = fold_number
            timerange_df = timerange_df.append(tmp_df)
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
        return forecasts

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
            Dataset to fit models in backtest
        metrics:
            List of metrics to compute for each fold
        n_folds:
            Number of folds
        mode:
            One of 'expand', 'constant' -- train generation policy
        aggregate_metrics:
            If True aggregate metrics above folds, return raw metrics otherwise
        n_jobs:
            Number of jobs to run in parallel
        joblib_params:
            Additional parameters for joblib.Parallel

        Returns
        -------
        metrics_df, forecast_df, fold_info_df:
            Metrics dataframe, forecast dataframe and dataframe with information about folds
        """
        self._init_backtest()
        self._validate_backtest_n_folds(n_folds=n_folds)
        self._validate_backtest_dataset(ts=ts, n_folds=n_folds, horizon=self.horizon)
        self._validate_backtest_metrics(metrics=metrics)
        folds = Parallel(n_jobs=n_jobs, **joblib_params)(
            delayed(self._run_fold)(train=train, test=test, fold_number=fold_number, metrics=metrics)
            for fold_number, (train, test) in enumerate(
                self._generate_folds_datasets(ts=ts, n_folds=n_folds, horizon=self.horizon, mode=mode)
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
