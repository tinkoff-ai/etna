import numpy as np
import pytest

from etna.transforms import BinsegTrendTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import STLTransform
from etna.transforms import TheilSenTrendTransform


@pytest.mark.parametrize(
    "transform",
    (
        BinsegTrendTransform(in_column="target", n_bkps=1, min_size=1, model="l2"),
        STLTransform(in_column="target", period=2),
        LinearTrendTransform(in_column="target"),
        TheilSenTrendTransform(in_column="target"),
    ),
)
def test_dummy_all(toy_dataset_equal_targets_and_quantiles, transform):
    """This test checks that inverse_transform transforms forecast's quantiles the same way with target itself."""
    toy_dataset = toy_dataset_equal_targets_and_quantiles.to_pandas()
    _ = transform.fit_transform(toy_dataset.copy())

    toy_dataset = transform.inverse_transform(toy_dataset)

    np.testing.assert_allclose(toy_dataset.iloc[:, 0], toy_dataset.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset.iloc[:, 2], toy_dataset.iloc[:, 3])
