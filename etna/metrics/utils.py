from typing import Dict
from typing import List
from typing import Union

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
    dict of metrics in format {"metric_name": metric_value}
    """
    metrics_values = {}
    for metric in metrics:
        metrics_values[metric.__repr__()] = metric(y_true=y_true, y_pred=y_pred)
    return metrics_values
