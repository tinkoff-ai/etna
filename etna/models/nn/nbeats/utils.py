from typing import Any
from typing import Dict
from typing import List
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
    return torch.tensor(x, dtype=torch.float32)


def prepare_train_batch(
    data: List[Dict[str, Any]],
    input_size: int,
    output_size: int,
    window_sampling_limit: Optional[int] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Dict[str, Optional["torch.Tensor"]]:
    """Prepare batch with training data."""
    if random_state is None:
        random_state = np.random.RandomState()

    batch_size = len(data)

    history = np.zeros((batch_size, input_size))
    history_mask = np.zeros((batch_size, input_size))
    target = np.zeros((batch_size, output_size))
    target_mask = np.zeros((batch_size, output_size))

    for i, part in enumerate(data):
        series = part["history"]

        if window_sampling_limit is not None:
            lower_bound = max(1, len(series) - window_sampling_limit - 1)

        else:
            lower_bound = 1

        cut_point = random_state.randint(low=lower_bound, high=len(series) - 1, size=1)[0]

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


def prepare_test_batch(data: List[Dict[str, Any]], input_size: int) -> Dict[str, Any]:
    """Prepare batch with data for forecasting."""
    batch_size = len(data)
    history = np.zeros((batch_size, input_size))
    history_mask = np.zeros((batch_size, input_size))
    segments = []

    for i, part in enumerate(data):
        series = part["history"]

        nan_mask = np.isnan(series)
        first_non_nan = int(np.argmin(nan_mask))
        last_non_nan = len(nan_mask) - int(np.argmin(nan_mask[::-1]))

        insample_window = series[max(last_non_nan - input_size, first_non_nan) : last_non_nan]

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
