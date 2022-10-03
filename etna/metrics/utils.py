from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

from etna.datasets import TSDataset
from etna.metrics import Metric


def compute_metrics(
    metrics: List[Metric], y_true: TSDataset, y_pred: TSDataset
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute metrics for given y_true, y_pred.

    Parameters
    ----------
    metrics:
        list of metrics to compute
    y_true:
        dataset of true values of time series
    y_pred:
        dataset of time series forecast
    Returns
    -------
    :
        dict of metrics in format {"metric_name": metric_value}
    """
    metrics_values = {}
    for metric in metrics:
        metrics_values[metric.__repr__()] = metric(y_true=y_true, y_pred=y_pred)
    return metrics_values


def percentile(n: int):
    """Percentile for pandas agg."""

    def percentile_(x):
        return np.nanpercentile(a=x.values, q=n)

    percentile_.__name__ = f"percentile_{n}"
    return percentile_


MetricAggregationStatistics = Literal[
    "median", "mean", "std", "percentile_5", "percentile_25", "percentile_75", "percentile_95"
]

METRICS_AGGREGATION_MAP: Dict[MetricAggregationStatistics, Union[str, Callable]] = {
    "median": "median",
    "mean": "mean",
    "std": "std",
    "percentile_5": percentile(5),
    "percentile_25": percentile(25),
    "percentile_75": percentile(75),
    "percentile_95": percentile(95),
}


def aggregate_metrics_df(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate metrics in :py:meth:`log_backtest_metrics` method.

    Parameters
    ----------
    metrics_df:
        Dataframe produced with :py:meth:`etna.pipeline.Pipeline._get_backtest_metrics`
    """
    # case for aggregate_metrics=False
    if "fold_number" in metrics_df.columns:
        metrics_dict = (
            metrics_df.groupby("segment")
            .mean()
            .reset_index()
            .drop(["segment", "fold_number"], axis=1)
            .apply(list(METRICS_AGGREGATION_MAP.values()))
            .to_dict()
        )

    # case for aggregate_metrics=True
    else:
        metrics_dict = metrics_df.drop(["segment"], axis=1).apply(list(METRICS_AGGREGATION_MAP.values())).to_dict()

    metrics_dict_wide = {
        f"{metrics_key}_{statistics_key}": value
        for metrics_key, values in metrics_dict.items()
        for statistics_key, value in values.items()
    }

    return metrics_dict_wide
