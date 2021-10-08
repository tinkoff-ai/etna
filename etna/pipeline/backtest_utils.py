from enum import Enum
from typing import Dict
from typing import List
from typing import Tuple

from etna.datasets import TSDataset
from etna.metrics import Metric


class CrossValidationMode(Enum):
    """Enum for different cross-validation modes."""

    expand = "expand"
    constant = "constant"


def generate_folds_datasets(
    ts: TSDataset, n_folds: int, horizon: int, mode: str = "expand"
) -> Tuple[TSDataset, TSDataset]:
    """Generate a sequence of train-test pairs according to timestamp.

    Parameters
    ----------
    ts:
        dataset to split
    n_folds:
        number of folds to split dataset to
    horizon:
        horizon length
    mode:
        ...

    Returns
    -------
    tuple of train and test dataset
    """
    mode = CrossValidationMode[mode.lower()]
    if mode == CrossValidationMode.expand:
        constant_history_length = 0
    elif mode == CrossValidationMode.constant:
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


def validate_backtest_dataset(ts: TSDataset, n_folds: int, horizon: int):
    """
    Check that all the given timestamps have enough timestamp points to validate forecaster with given number of splits.

    Parameters
    ----------
    ts:
        dataset to validate

    Raises
    ------
    ValueError:
        if there is no enough timestamp points to validate forecaster
    """
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


def compute_metrics(metrics: List[Metric], y_true: TSDataset, y_pred: TSDataset) -> Dict[str, float]:
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
        metrics_values[metric.__class__.__name__] = metric(y_true=y_true, y_pred=y_pred)
    return metrics_values
