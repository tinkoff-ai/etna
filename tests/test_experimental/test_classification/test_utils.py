import numpy as np
import pytest

from etna.experimental.classification.utils import crop_nans_single_series
from etna.experimental.classification.utils import padd_single_series


def test_crop_nans_single_series(
    x=np.array([None, None, 1, 2, 3], dtype=float), x_expected=np.array([1, 2, 3], dtype=float)
):
    x_cropped = crop_nans_single_series(x)
    np.testing.assert_array_equal(x_cropped, x_expected)


def test_crop_nans_single_series_raise_error_all_nans(x=np.array([None, None, None], dtype=float)):
    with pytest.raises(ValueError, match="Dataset contains the series all consists of NaN values!"):
        _ = crop_nans_single_series(x)


@pytest.mark.parametrize(
    "padding_value, expected_array",
    [(0, np.array([0, 0, 1, 2, 3, 4, 5])), ("back_fill", np.array([1, 1, 1, 2, 3, 4, 5]))],
)
def test_transform_single_series(padding_value, expected_array, expected_len=7, x=np.array([1, 2, 3, 4, 5])):
    transformed_series = padd_single_series(x=x, expected_len=expected_len, padding_value=padding_value)
    assert np.all(transformed_series == expected_array)
