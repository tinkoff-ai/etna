import numpy as np
import pytest

from etna.experimental.classification.utils import crop_nans_single_series


def test_crop_nans_single_series(
    x=np.array([None, None, 1, 2, 3], dtype=float), x_expected=np.array([1, 2, 3], dtype=float)
):
    x_cropped = crop_nans_single_series(x)
    np.testing.assert_array_equal(x_cropped, x_expected)


def test_crop_nans_single_series_raise_error_all_nans(x=np.array([None, None, None], dtype=float)):
    with pytest.raises(ValueError, match="Dataset contains the series all consists of NaN values!"):
        _ = crop_nans_single_series(x)
