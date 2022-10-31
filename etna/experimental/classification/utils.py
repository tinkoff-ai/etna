from typing import Union

import numpy as np
from typing_extensions import Literal


def crop_nans_single_series(x: np.ndarray) -> np.ndarray:
    """Crop the trailing nans from the single series."""
    if np.sum(~np.isnan(x)) == 0:
        raise ValueError("Dataset contains the series all consists of NaN values!")
    first_non_nan_ind = np.where(~np.isnan(x))[0][0]
    x = x[first_non_nan_ind:]
    return x


def padd_single_series(
    x: np.ndarray, expected_len: int, padding_value: Union[float, Literal["back_fill"]]
) -> np.ndarray:
    """Apply padding on a single series."""
    x = x[-expected_len:]
    history_len = len(x)
    padding_len = expected_len - history_len
    padding_value = x[0] if padding_value == "back_fill" else padding_value
    x = np.pad(array=x, pad_width=(padding_len, 0), mode="constant", constant_values=padding_value)
    return x
