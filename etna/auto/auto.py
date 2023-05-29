import itertools
import time
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from hydra_slayer import get_from_params
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler
from optuna.storages import BaseStorage
from optuna.storages import RDBStorage
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from typing_extensions import Protocol

from etna.auto.optuna import ConfigSampler
from etna.auto.optuna import Optuna
from etna.auto.pool import Pool
from etna.auto.runner import AbstractRunner
from etna.auto.runner import LocalRunner
from etna.auto.utils import config_hash
from etna.datasets import TSDataset
from etna.loggers import tslogger
from etna.metrics import MAE
from etna.metrics import MSE
from etna.metrics import SMAPE
from etna.metrics import MedAE
from etna.metrics import Metric
from etna.metrics import Sign
from etna.metrics.utils import MetricAggregationStatistics
from etna.metrics.utils import aggregate_metrics_df
from etna.pipeline.base import BasePipeline


class _Callback(Protocol):
    def __call__(self, metrics_df: pd.DataFrame, forecast_df: pd.DataFrame, fold_info_df: pd.DataFrame) -> None:
        ...


class _Initializer(Protocol):
    def __call__(self, pipeline: BasePipeline) -> None:
        ...


class AutoAbstract(ABC):
    """Interface for ``Auto`` object."""

    @abstractmethod
    def fit(
        self,
        ts: TSDataset,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        initializer: Optional[_Initializer] = None,
        callback: Optional[_Callback] = None,
        optuna_params: Optional[Dict[str, Any]] = None,
    ) -> BasePipeline:
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
        optuna_params:
            additional kwargs for optuna :py:meth:`optuna.study.Study.optimize`
        """
        pass

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Get Auto trials summary."""
        pass

    @abstractmethod
    def top_k(self, k: int = 5) -> List[BasePipeline]:
        """
        Get top k pipelines with the best metric value.

        Parameters
        ----------
        k:
            number of pipelines to return
        """
        pass


class AutoBase(AutoAbstract):
    """Base Class for ``Auto`` and ``Tune``, implementing core logic behind these classes."""

    def __init__(
        self,
        target_metric: Metric,
        horizon: int,
        metric_aggregation: MetricAggregationStatistics = "mean",
        backtest_params: Optional[dict] = None,
        experiment_folder: Optional[str] = None,
        runner: Optional[AbstractRunner] = None,
        storage: Optional[BaseStorage] = None,
        metrics: Optional[List[Metric]] = None,
    ):
        """
        Initialize AutoBase class.

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
            name for saving experiment results, it determines the name for optuna study
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
        self.horizon = horizon
        self.metric_aggregation: MetricAggregationStatistics = metric_aggregation
        self.backtest_params = {} if backtest_params is None else backtest_params
        self.experiment_folder = experiment_folder

        self.runner = LocalRunner() if runner is None else runner
        self.storage = RDBStorage("sqlite:///etna-auto.db") if storage is None else storage

        metrics = [Sign(), SMAPE(), MAE(), MSE(), MedAE()] if metrics is None else metrics
        if str(target_metric) not in [str(metric) for metric in metrics]:
            metrics.append(target_metric)
        self.metrics = metrics

    def _top_k(self, summary: pd.DataFrame, k: int) -> List[BasePipeline]:
        df = summary.sort_values(
            by=[f"{self.target_metric.name}_{self.metric_aggregation}"],
            ascending=(not self.target_metric.greater_is_better),
        )
        return [pipeline for pipeline in df["pipeline"].values[:k]]  # noqa: C416

    def top_k(self, k: int = 5) -> List[BasePipeline]:
        """
        Get top k pipelines with the best metric value.

        Parameters
        ----------
        k:
            number of pipelines to return
        """
        summary = self.summary()
        return self._top_k(summary=summary, k=k)


class Auto(AutoBase):
    """Automatic pipeline selection via defined or custom pipeline pool."""

    def __init__(
        self,
        target_metric: Metric,
        horizon: int,
        metric_aggregation: MetricAggregationStatistics = "mean",
        backtest_params: Optional[dict] = None,
        experiment_folder: Optional[str] = None,
        pool: Union[Pool, List[BasePipeline]] = Pool.default,
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
            name for saving experiment results, it determines the name for optuna study
        pool:
            pool of pipelines to choose from
        runner:
            runner to use for distributed training
        storage:
            optuna storage to use
        metrics:
            list of metrics to compute
        """
        super().__init__(
            target_metric=target_metric,
            horizon=horizon,
            metric_aggregation=metric_aggregation,
            backtest_params=backtest_params,
            experiment_folder=experiment_folder,
            runner=runner,
            storage=storage,
            metrics=metrics,
        )
        self.pool = pool
        self._pool = self._make_pool(pool=pool, horizon=horizon)
        self._pool_optuna: Optional[Optuna] = None

        root_folder = self.experiment_folder if self.experiment_folder is not None else ""
        self._pool_folder = f"{root_folder}/pool"

    @staticmethod
    def _make_pool(pool: Union[Pool, List[BasePipeline]], horizon: int) -> List[BasePipeline]:
        if isinstance(pool, Pool):
            list_pool: List[BasePipeline] = list(pool.value.generate(horizon=horizon))
        else:
            list_pool = list(pool)

        return list_pool

    def _get_tuner_timeout(self, timeout: Optional[int], tune_size: int, elapsed_time: float) -> Optional[int]:
        if timeout is None or tune_size < 1:
            return None
        else:
            tune_timeout = (timeout - elapsed_time) // tune_size
            return int(tune_timeout)

    def _get_tuner_n_trials(self, n_trials: Optional[int], tune_size: int, elapsed_n_trials) -> Optional[int]:
        if n_trials is None or tune_size < 1:
            return None
        else:
            tune_n_trials = (n_trials - elapsed_n_trials) // tune_size
            return tune_n_trials

    def _fit_tuner(
        self,
        pipeline: BasePipeline,
        ts: TSDataset,
        timeout: Optional[int],
        n_trials: Optional[int],
        initializer: Optional[_Initializer],
        callback: Optional[_Callback],
        folder: str,
        optuna_params: Dict[str, Any],
    ) -> "Tune":
        cur_tuner = Tune(
            pipeline=pipeline,
            target_metric=self.target_metric,
            horizon=self.horizon,
            metric_aggregation=self.metric_aggregation,
            backtest_params=self.backtest_params,
            experiment_folder=folder,
            runner=self.runner,
            storage=self.storage,
            metrics=self.metrics,
            sampler=None,
        )
        _ = cur_tuner.fit(
            ts=ts,
            timeout=timeout,
            n_trials=n_trials,
            initializer=initializer,
            callback=callback,
            optuna_params=optuna_params,
        )
        return cur_tuner

    def _get_tune_folder(self, pipeline: BasePipeline) -> str:
        config = pipeline.to_dict()
        identifier = config_hash(config)
        root_folder = self.experiment_folder if self.experiment_folder is not None else ""
        folder = f"{root_folder}/tuning/{identifier}"
        return folder

    def _top_k_pool(self, k: int):
        if self._pool_optuna is None:
            self._pool_optuna = self._init_pool_optuna()

        pool_trials = self._pool_optuna.study.get_trials()
        pool_summary = self._make_pool_summary(trials=pool_trials)
        df = pd.DataFrame(pool_summary)
        return self._top_k(summary=df, k=k)

    def fit(
        self,
        ts: TSDataset,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        initializer: Optional[_Initializer] = None,
        callback: Optional[_Callback] = None,
        optuna_params: Optional[Dict[str, Any]] = None,
        tune_size: int = 0,
    ) -> BasePipeline:
        """
        Start automatic pipeline selection.

        There are two stages:

        - Pool-stage: trying every pipeline in a pool
        - Tuning-stage: tuning `tune_size` best pipelines from a previous stage by using :py:class`~etna.auto.auto.Tune`.

        Tuning stage starts only if limits on `n_trials` and `timeout` aren't exceeded.
        Tuning goes from the best pipeline to the worst, and
        trial limits (`n_trials`, `timeout`) are divided evenly between each pipeline.
        If there are no limits on number of trials only the first pipeline will be tuned until user stops the process.

        Parameters
        ----------
        ts:
            tsdataset to fit on
        timeout:
            timeout for optuna. N.B. this is timeout for each worker
        n_trials:
            number of trials for optuna. N.B. this is number of trials for each worker
        tune_size:
            how many best pipelines to tune after checking the pool
        initializer:
            is called before each pipeline backtest, can be used to initialize loggers
        callback:
            is called after each pipeline backtest, can be used to log extra metrics
        optuna_params:
            additional kwargs for optuna :py:meth:`optuna.study.Study.optimize`
        """
        if optuna_params is None:
            optuna_params = {}

        if self._pool_optuna is None:
            self._pool_optuna = self._init_pool_optuna()

        tslogger.log("Trying pipelines from a pool")
        start_pool_tuning_time = time.perf_counter()
        self._pool_optuna.tune(
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
            **optuna_params,
        )

        pool_elapsed_time = time.perf_counter() - start_pool_tuning_time
        pool_elapsed_n_trials = len(self._pool_optuna.study.get_trials())
        tuner_timeout = self._get_tuner_timeout(timeout=timeout, tune_size=tune_size, elapsed_time=pool_elapsed_time)
        tuner_n_trials = self._get_tuner_n_trials(
            n_trials=n_trials, tune_size=tune_size, elapsed_n_trials=pool_elapsed_n_trials
        )
        if (tuner_n_trials is None or tuner_n_trials > 0) and (tuner_timeout is None or tuner_timeout > 0):
            tslogger.log("Tuning best pipelines from the pool")
            best_pool_pipelines = self._top_k_pool(k=tune_size)
            for i in range(tune_size):
                cur_pipeline = best_pool_pipelines[i]
                cur_folder = self._get_tune_folder(cur_pipeline)
                tslogger.log(f"Tuning top-{i+1} pipeline")
                _ = self._fit_tuner(
                    pipeline=cur_pipeline,
                    ts=ts,
                    timeout=tuner_timeout,
                    n_trials=tuner_n_trials,
                    initializer=initializer,
                    callback=callback,
                    folder=cur_folder,
                    optuna_params=optuna_params,
                )

        best_pipeline = self.top_k(k=1)[0]
        return best_pipeline

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
        Optuna objective wrapper for the pool stage.

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

        Returns
        -------
        objective:
            function that runs specified trial and returns its evaluated score
        """

        def _objective(trial: Trial) -> float:

            pipeline_config = dict()
            pipeline_config.update(trial.relative_params)
            pipeline_config.update(trial.params)

            pipeline: BasePipeline = get_from_params(**pipeline_config)
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

    def _init_pool_optuna(self) -> Optuna:
        """Initialize optuna."""
        pool = [pipeline.to_dict() for pipeline in self._pool]
        pool_optuna = Optuna(
            direction="maximize" if self.target_metric.greater_is_better else "minimize",
            study_name=self._pool_folder,
            storage=self.storage,
            sampler=ConfigSampler(configs=pool),
        )
        return pool_optuna

    def _init_tuners(self, pool_optuna: Optuna) -> List["Tune"]:
        trials = pool_optuna.study.get_trials()
        configs = [trial.user_attrs["pipeline"] for trial in trials]

        results = []
        for config in configs:
            cur_pipeline = get_from_params(**config)
            cur_folder = self._get_tune_folder(cur_pipeline)
            tuner = Tune(
                pipeline=cur_pipeline,
                target_metric=self.target_metric,
                horizon=self.horizon,
                metric_aggregation=self.metric_aggregation,
                backtest_params=self.backtest_params,
                experiment_folder=cur_folder,
                runner=self.runner,
                storage=self.storage,
                metrics=self.metrics,
                sampler=None,
            )
            results.append(tuner)

        return results

    def _make_pool_summary(self, trials: List[FrozenTrial]) -> List[dict]:
        """Get information from trial summary."""
        study_params = [
            {
                **trial.user_attrs,
                "study": self._pool_folder,
                "pipeline": get_from_params(**trial.user_attrs["pipeline"]),
                "state": trial.state,
            }
            for trial in trials
        ]
        return study_params

    def _make_tune_summary(self, trials: List[FrozenTrial], pipeline: BasePipeline) -> List[dict]:
        """Get information from trial summary."""
        study = self._get_tune_folder(pipeline)
        study_params = [
            {
                **trial.user_attrs,
                "study": study,
                "pipeline": get_from_params(**trial.user_attrs["pipeline"]),
                "state": trial.state,
            }
            for trial in trials
        ]
        return study_params

    def summary(self) -> pd.DataFrame:
        """Get Auto trials summary.

        Returns
        -------
        study_dataframe:
            dataframe with detailed info on each performed trial
        """
        if self._pool_optuna is None:
            self._pool_optuna = self._init_pool_optuna()

        pool_trials = self._pool_optuna.study.get_trials()
        pool_summary = self._make_pool_summary(trials=pool_trials)

        tuners = self._init_tuners(self._pool_optuna)
        tune_pipelines = [t.pipeline for t in tuners]
        tune_trials = [t._init_optuna().study.get_trials() for t in tuners]
        tune_summary = [self._make_tune_summary(trials=t, pipeline=p) for t, p in zip(tune_trials, tune_pipelines)]

        total_summary = pool_summary + list(itertools.chain(*tune_summary))
        return pd.DataFrame(total_summary)


class Tune(AutoBase):
    """Automatic tuning of custom pipeline.

    This class takes given pipelines and tries to optimize its hyperparameters by using `params_to_tune`.
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        target_metric: Metric,
        horizon: int,
        metric_aggregation: MetricAggregationStatistics = "mean",
        backtest_params: Optional[dict] = None,
        experiment_folder: Optional[str] = None,
        runner: Optional[AbstractRunner] = None,
        storage: Optional[BaseStorage] = None,
        metrics: Optional[List[Metric]] = None,
        sampler: Optional[BaseSampler] = None,
    ):
        """
        Initialize Tune class.

        Parameters
        ----------
        pipeline:
            pipeline to optimize
        target_metric:
            metric to optimize
        horizon:
            horizon to forecast for
        metric_aggregation:
            aggregation method for per-segment metrics
        backtest_params:
            custom parameters for backtest instead of default backtest parameters
        experiment_folder:
            name for saving experiment results, it determines the name for optuna study
        runner:
            runner to use for distributed training
        storage:
            optuna storage to use
        metrics:
            list of metrics to compute
        """
        super().__init__(
            target_metric=target_metric,
            horizon=horizon,
            metric_aggregation=metric_aggregation,
            backtest_params=backtest_params,
            experiment_folder=experiment_folder,
            runner=runner,
            storage=storage,
            metrics=metrics,
        )
        self.pipeline = pipeline
        if sampler is None:
            self.sampler: BaseSampler = TPESampler()
        else:
            self.sampler = sampler
        self._optuna: Optional[Optuna] = None

    def fit(
        self,
        ts: TSDataset,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        initializer: Optional[_Initializer] = None,
        callback: Optional[_Callback] = None,
        optuna_params: Optional[Dict[str, Any]] = None,
    ) -> BasePipeline:
        """
        Start automatic pipeline tuning.

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
        optuna_params:
            additional kwargs for optuna :py:meth:`optuna.study.Study.optimize`
        """
        if optuna_params is None:
            optuna_params = {}

        if self._optuna is None:
            self._optuna = self._init_optuna()

        self._optuna.tune(
            objective=self.objective(
                ts=ts,
                pipeline=self.pipeline,
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
            **optuna_params,
        )

        return get_from_params(**self._optuna.study.best_trial.params)

    @staticmethod
    def objective(
        ts: TSDataset,
        pipeline: BasePipeline,
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
        pipeline:
            pipeline to tune
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

        Returns
        -------
        objective:
            function that runs specified trial and returns its evaluated score
        """
        dict_of_distrs = {
            UniformDistribution: lambda x: ("suggest_uniform", {"low": x.low, "high": x.high}),
            LogUniformDistribution: lambda x: ("suggest_loguniform", {"low": x.low, "high": x.high}),
            DiscreteUniformDistribution: lambda x: (
                "suggest_discrete_uniform",
                {"low": x.low, "high": x.high, "q": x.q},
            ),
            IntUniformDistribution: lambda x: ("suggest_int", {"low": x.low, "high": x.high, "step": x.step}),
            IntLogUniformDistribution: lambda x: (
                "suggest_int",
                {"low": x.low, "high": x.high, "step": x.step},
            ),
            CategoricalDistribution: lambda x: ("suggest_categorical", {"choices": x.choices}),
        }

        def _objective(trial: Trial) -> float:

            params_to_tune = pipeline.params_to_tune()

            # using received optuna.distribution objects to call corresponding trial.suggest_xxx
            params_suggested = {}
            for param_name, param_distr in params_to_tune.items():
                method_name, method_kwargs = dict_of_distrs[type(param_distr)](param_distr)
                method = getattr(trial, method_name)
                params_suggested[param_name] = method(param_name, **method_kwargs)

            # create pipeline instance with the parameters to try
            pipeline_trial_params: BasePipeline = pipeline.set_params(**params_suggested)

            if initializer is not None:
                initializer(pipeline=pipeline_trial_params)

            metrics_df, forecast_df, fold_info_df = pipeline_trial_params.backtest(
                ts, metrics=metrics, **backtest_params
            )

            if callback is not None:
                callback(metrics_df=metrics_df, forecast_df=forecast_df, fold_info_df=fold_info_df)

            trial.set_user_attr("pipeline", pipeline_trial_params.to_dict())

            aggregated_metrics = aggregate_metrics_df(metrics_df)
            for metric in aggregated_metrics:
                trial.set_user_attr(metric, aggregated_metrics[metric])

            return aggregated_metrics[f"{target_metric.name}_{metric_aggregation}"]

        return _objective

    def _init_optuna(self) -> Optuna:
        """Initialize optuna."""
        # sampler receives no hyperparameters here and optimizes only the hyperparameters suggested in objective
        optuna = Optuna(
            direction="maximize" if self.target_metric.greater_is_better else "minimize",
            study_name=self.experiment_folder,
            storage=self.storage,
            sampler=self.sampler,
        )
        return optuna

    def _summary(self, trials: List[FrozenTrial]) -> List[dict]:
        """Get information from trial summary."""
        study_params = [
            {**trial.user_attrs, "pipeline": get_from_params(**trial.user_attrs["pipeline"]), "state": trial.state}
            for trial in trials
        ]
        return study_params

    def summary(self) -> pd.DataFrame:
        """Get Auto trials summary.

        Returns
        -------
        study_dataframe:
            dataframe with detailed info on each performed trial
        """
        if self._optuna is None:
            self._optuna = self._init_optuna()

        trials = self._optuna.study.get_trials()

        study_params = self._summary(trials=trials)
        return pd.DataFrame(study_params)
