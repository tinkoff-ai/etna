
from typing_extensions import Literal



class Auto:

  def __init__(
    self,
    metric: Metric,
    metric_aggregation: Literal['mean', 'median'], # metrics aggregates like we make in loggers for summary metrics https://github.com/tinkoff-ai/etna/blob/0198d11943db4cfe8ad74632f6850d54f2f7a772/etna/loggers/base.py#L202
    backtest_params: Dict,
    horizon: int,
    experiment_folder: str, #used as study name for now
    pool: Optional[Pool, List[Pipeline]] = Pool.default,
    runner: Runner = LocalRunner,
    storage: optuna.BaseStorage = optuna.storages.RDBStorage(#sqlite db:etna-auto),
  ):
    pass
  
  def fit(
    self,
    ts: TSDataset,
    timeout: Optional[float] = None,
    n_trials: Optional[int] = None,
    initializer: Optional[Callable[Pipeline, None]] = None,
    callback: Optional[Callable[[DataFrame, DataFrame, DataFrame], None]] = None, # backtest output
    **optuna_kwargs,
  ) -> Pipeline:
  
  def top_k(
    self,
    n_best: 5
  ) -> List[Pipeline]
    pass
  
  def runs_result(self) -> DataFrame:
    # returns: | etna.pipeline.Pipeline | agg_metric_1 | agg_metric_2 | ... | agg_metric_N |
    pass
  
  @staticmethod
  def objective(
        ts: TSDataset,
        metric: Metric,
        metric_aggregation: Literal['mean', 'median'], 
        backtest_params: dict,
        callback: Optional[Callable[Pipeline, None]] = None,
        initializer: Optional[Callable[[DataFrame, DataFrame, DataFrame], None]] = None,
    ) -> Callable[optuna.trial.Trail, float]
       """ Return oputna like objective with bactkest running and calling `initializer`, `callback`  functions. 
            We compute all metrics in backtest except R2 because it's ill-posed
       """