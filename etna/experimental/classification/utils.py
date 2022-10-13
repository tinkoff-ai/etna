import numpy as np


def crop_nans_single_series(x: np.ndarray) -> np.ndarray:
    """Crop the trailing nans from the single series."""
    if np.sum(~np.isnan(x)) == 0:
        raise ValueError("Dataset contains the series all consists of NaN values!")
    first_non_nan_ind = np.where(~np.isnan(x))[0][0]
    x = x[first_non_nan_ind:]
    return x
