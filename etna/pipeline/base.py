import warnings
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Sequence, List, Any, Dict, Tuple, Optional, Generator

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.stats import norm

from etna.core import BaseMixin
from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE, Metric, MetricAggregationMode


class CrossValidationMode(Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"


class PipelineBase(ABC, BaseMixin):
    @abstractmethod
    def __init__(self):
        self.horizon = None
        self.n_folds_interval = None
        self.quantiles = None
        self.ts = None
        self._folds = None

    @abstractmethod
    def fit(self, ts: TSDataset) -> "PipelineBase":
        pass

    @abstractmethod
    def forecast(self, prediction_interval: bool = False) -> TSDataset:
        pass

    @staticmethod
    def _validate_horizon(horizon: int) -> int:
        """Check that given number of folds is grater than 1."""
        if horizon > 0:
            return horizon
        else:
            raise ValueError("At least one point in the future is expected.")

    @property
    @abstractmethod
    def support_prediction_interval(self) -> bool:
        """Indicate if method supports prediction intervals."""
        pass

    def check_support_prediction_interval(self, prediction_interval_option: bool = False):
        """Check if pipeline supports prediction intervals, if not, warns a user.

        Parameters
        ----------
        prediction_interval_option:
            indicate if forecast method is called with `prediction_interval=True`
        """
        if not self.support_prediction_interval and prediction_interval_option:
            warnings.warn("This class doesn't support prediction intervals and they won't be build")

    def backtest(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented")


class IntervalMixin(PipelineBase, ABC):
    @staticmethod
    def _validate_quantiles(quantiles: Sequence[float]) -> Sequence[float]:
        """Check that given number of folds is grater than 1."""
        for quantile in quantiles:
            if not (0 < quantile < 1):
                raise ValueError("Quantile should be a number from (0,1).")
        return quantiles

    def _forecast_prediction_interval(self, predictions: TSDataset) -> TSDataset:
        """Forecast prediction interval for the future."""
        if self.ts is None:
            raise ValueError("Pipeline is not fitted! Fit the Pipeline before calling forecast method.")
        _, forecasts, _ = self.backtest(self.ts, metrics=[MAE()], n_folds=self.n_folds_interval)
        forecasts = TSDataset(df=forecasts, freq=self.ts.freq)
        residuals = (
                forecasts.loc[:, pd.IndexSlice[:, "target"]]
                - self.ts[forecasts.index.min(): forecasts.index.max(), :, "target"]
        )
        se = scipy.stats.sem(residuals)
        borders = []
        for quantile in self.quantiles:
            z_q = norm.ppf(q=quantile)
            border = predictions[:, :, "target"] + se * z_q
            border.rename({"target": f"target_{quantile:.4g}"}, inplace=True, axis=1)
            borders.append(border)

        predictions.df = pd.concat([predictions.df] + borders, axis=1).sort_index(axis=1, level=(0, 1))

        return predictions


class BacktestMixin(PipelineBase, ABC):
    def backtest(
            self,
            ts: TSDataset,
            metrics: List[Metric],
            n_folds: int = 5,
            mode: str = "expand",
            aggregate_metrics: bool = False,
            n_jobs: int = 1,
            joblib_params: Dict[str, Any] = None,
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
        pd.DataFrame, pd.DataFrame, pd.Dataframe:
            metrics dataframe, forecast dataframe and dataframe with information about folds
        """
        self._init_backtest()
        self._validate_backtest_n_folds(n_folds=n_folds)
        self._validate_backtest_dataset(ts=ts, n_folds=n_folds, horizon=self.horizon)
        self._validate_backtest_metrics(metrics=metrics)
        joblib_params = joblib_params if joblib_params is not None \
            else dict(verbose=11, backend="multiprocessing", mmap_mode="c")
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
                    f"{metric.__class__.__name__} metric is in {metric.mode} mode"
                )

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

    @staticmethod
    def _compute_metrics(metrics: List[Metric], y_true: TSDataset, y_pred: TSDataset) -> Dict[str, Dict[str, float]]:
        """Compute metrics for given y_true, y_pred."""
        metrics_values: Dict[str, Dict[str, float]] = {}
        for metric in metrics:
            metrics_values[metric.__class__.__name__] = metric(y_true=y_true, y_pred=y_pred)  # type: ignore
        return metrics_values

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