from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import numpy as np

from etna import SETTINGS

if SETTINGS.torch_required:
    import torch


def _create_or_update(param: Optional[Dict], name: str, value: Any):
    """Create new parameters dict or add field to existing one."""
    if param is None:
        param = {name: value}
    else:
        param[name] = value
    return param


def default_torch_device() -> "torch.device":
    """Return CUDA device if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x: Any) -> "torch.Tensor":
    """Convert data to tensor and put on default device.

    Parameters
    ----------
    x:
        Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.

    Returns
    -------
    :
        Input data as tensor.
    """
    return torch.tensor(x, dtype=torch.float32).to(default_torch_device())


def prepare_train_batch(
    data: Iterable[Dict[str, Any]], batch_size: int, input_size: int, output_size: int
) -> Dict[str, Optional["torch.Tensor"]]:
    """Prepare batch with training data."""
    history = np.zeros((batch_size, input_size))
    history_mask = np.zeros((batch_size, input_size))
    target = np.zeros((batch_size, output_size))
    target_mask = np.zeros((batch_size, output_size))

    for i, part in enumerate(data):
        series = part["history"]
        cut_point = np.random.randint(low=1, high=len(series) - 1, size=1)[0]

        insample_window = series[max(0, cut_point - input_size) : cut_point]
        history[i, -len(insample_window) :] = insample_window
        history_mask[i, -len(insample_window) :] = 1.0

        outsample_window = series[cut_point : min(len(series), cut_point + output_size)]
        target[i, : len(outsample_window)] = outsample_window
        target_mask[i, : len(outsample_window)] = 1.0

    batch = {
        "history": to_tensor(history),
        "history_mask": to_tensor(history_mask),
        "target": to_tensor(target),
        "target_mask": to_tensor(target_mask),
        "segment": None,
    }

    return batch


def prepare_test_batch(data: Iterable[Dict[str, Any]], batch_size: int, input_size: int) -> Dict[str, Any]:
    """Prepare batch with data for forecasting."""
    history = np.zeros((batch_size, input_size))
    history_mask = np.zeros((batch_size, input_size))
    segments = []

    for i, part in enumerate(data):
        series = part["history"]

        first_non_nan = int(np.argmin(np.isnan(series)))
        insample_window = series[max(len(series) - input_size, first_non_nan) :]

        history[i, -len(insample_window) :] = insample_window
        history_mask[i, -len(insample_window) :] = 1.0
        segments.append(part["segment"])

    batch = {
        "history": to_tensor(history),
        "history_mask": to_tensor(history_mask),
        "target": None,
        "target_mask": None,
        "segment": segments,
    }
    return batch
