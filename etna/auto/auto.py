from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from hydra_slayer import get_from_params
from optuna.storages import BaseStorage
from optuna.storages import RDBStorage
from optuna.trial import Trial
from typing_extensions import Protocol

from etna.auto.optuna import ConfigSampler
from etna.auto.optuna import Optuna
from etna.auto.pool import Pool
from etna.auto.runner import AbstractRunner
from etna.auto.runner import LocalRunner
from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import MedAE
from etna.metrics import Metric
from etna.metrics import Sign
from etna.metrics.utils import MetricAggregationStatistics
from etna.metrics.utils import aggregate_metrics_df
from etna.pipeline import Pipeline


class _Callback(Protocol):
    def __call__(self, metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame) -> None:
        ...


class _Initializer(Protocol):
    def __call__(self, pipeline: Pipeline) -> None:
        ...


class Auto:
    """Automatic pipeline selection via defined or custom pipeline pool."""

    def __init__(
        self,
        target_metric: Metric,
        horizon: int,
        metric_aggregation: MetricAggregationStatistics = "mean",
        backtest_params: Optional[dict] = None,
        experiment_folder: Optional[str] = None,
        pool: Union[Pool, List[Pipeline]] = Pool.default,
        runner: Optional[AbstractRunner] = None,
        storage: Optional[BaseStorage] = None,
        metrics: Optional[List[Metric]] = None,
    ):
        """
        Initialize Auto class.

        Parameters
        ----------
        target_metric:
            metric to optimize
        horizon:
            horizon to forecast for
        metric_aggregation:
            aggregation method for per-segment metrics
        backtest_params:
            custom parameters for backtest instead of default backtest parameters
        experiment_folder:
            folder to store experiment results and name for optuna study
        pool:
            pool of pipelines to choose from
        runner:
            runner to use for distributed training
        storage:
            optuna storage to use
        metrics:
            list of metrics to compute
        """
        if target_metric.greater_is_better is None:
            raise ValueError("target_metric.greater_is_better is None")
        self.target_metric = target_metric

        self.metric_aggregation = metric_aggregation
        self.backtest_params = {} if backtest_params is None else backtest_params
        self.horizon = horizon
        self.experiment_folder = experiment_folder
        self.pool = pool
        self.runner = LocalRunner() if runner is None else runner
        self.storage = RDBStorage("sqlite:///etna-auto.db") if storage is None else storage

        metrics = [Sign(), SMAPE(), MAE(), MSE(), MedAE()] if metrics is None else metrics
        if str(target_metric) not in [str(metric) for metric in metrics]:
            metrics.append(target_metric)
        self.metrics = metrics
        self._optuna: Optional[Optuna] = None

    def fit(
        self,
        ts: TSDataset,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        initializer: Optional[_Initializer] = None,
        callback: Optional[_Callback] = None,
        **optuna_kwargs,
    ) -> Pipeline:
        """
        Start automatic pipeline selection.

        Parameters
        ----------
        ts:
            tsdataset to fit on
        timeout:
            timeout for optuna. N.B. this is timeout for each worker
        n_trials:
            number of trials for optuna. N.B. this is number of trials for each worker
        initializer:
            is called before each pipeline backtest, can be used to initialize loggers
        callback:
            is called after each pipeline backtest, can be used to log extra metrics
        optuna_kwargs:
            additional kwargs for optuna :py:meth:`optuna.study.Study.optimize`
        """
        if self._optuna is None:
            self._optuna = self._init_optuna()

        self._optuna.tune(
            objective=self.objective(
                ts=ts,
                target_metric=self.target_metric,
                metric_aggregation=self.metric_aggregation,
                metrics=self.metrics,
                backtest_params=self.backtest_params,
                initializer=initializer,
                callback=callback,
            ),
            runner=self.runner,
            n_trials=n_trials,
            timeout=timeout,
            **optuna_kwargs,
        )

        return get_from_params(**self._optuna.study.best_trial.user_attrs["pipeline"])

    def _init_optuna(self):
        """Initialize optuna."""
        if isinstance(self.pool, Pool):
            pool: List[Pipeline] = self.pool.value.generate(horizon=self.horizon)
        else:
            pool = self.pool

        pool_ = [pipeline.to_dict() for pipeline in pool]

        optuna = Optuna(
            direction="maximize" if self.target_metric.greater_is_better else "minimize",
            study_name=self.experiment_folder,
            storage=self.storage,
            sampler=ConfigSampler(configs=pool_),
        )
        return optuna

    def summary(self) -> pd.DataFrame:
        """Get Auto trials summary."""
        if self._optuna is None:
            self._optuna = self._init_optuna()

        study = self._optuna.study.get_trials()

        study_params = [
            {**trial.user_attrs, "pipeline": get_from_params(**trial.user_attrs["pipeline"]), "state": trial.state}
            for trial in study
        ]

        return pd.DataFrame(study_params)

    def top_k(self, k: int = 5) -> List[Pipeline]:
        """
        Get top k pipelines.

        Parameters
        ----------
        k:
            number of pipelines to return
        """
        summary = self.summary()
        df = summary.sort_values(
            by=[f"{self.target_metric.name}_{self.metric_aggregation}"],
            ascending=(not self.target_metric.greater_is_better),
        )
        return [pipeline for pipeline in df["pipeline"].values[:k]]  # noqa: C416

    @staticmethod
    def objective(
        ts: TSDataset,
        target_metric: Metric,
        metric_aggregation: MetricAggregationStatistics,
        metrics: List[Metric],
        backtest_params: dict,
        initializer: Optional[_Initializer] = None,
        callback: Optional[_Callback] = None,
    ) -> Callable[[Trial], float]:
        """
        Optuna objective wrapper.

        Parameters
        ----------
        ts:
            tsdataset to fit on
        target_metric:
            metric to optimize
        metric_aggregation:
            aggregation method for per-segment metrics
        metrics:
            list of metrics to compute
        backtest_params:
            custom parameters for backtest instead of default backtest parameters
        initializer:
            is called before each pipeline backtest, can be used to initialize loggers
        callback:
            is called after each pipeline backtest, can be used to log extra metrics
        """

        def _objective(trial: Trial) -> float:

            pipeline_config = dict()
            pipeline_config.update(trial.relative_params)
            pipeline_config.update(trial.params)

            pipeline: Pipeline = get_from_params(**pipeline_config)
            if initializer is not None:
                initializer(pipeline=pipeline)

            metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts, metrics=metrics, **backtest_params)

            if callback is not None:
                callback(metrics_df=metrics_df, forecast_df=forecast_df, fold_info_df=fold_info_df)

            aggregated_metrics = aggregate_metrics_df(metrics_df)

            for metric in aggregated_metrics:
                trial.set_user_attr(metric, aggregated_metrics[metric])

            return aggregated_metrics[f"{target_metric.name}_{metric_aggregation}"]

        return _objective
