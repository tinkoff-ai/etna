from copy import deepcopy

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
    toy_dataset = toy_dataset_equal_targets_and_quantiles
    transform.fit_transform(deepcopy(toy_dataset))

    toy_dataset_df = transform.inverse_transform(toy_dataset).to_pandas()

    np.testing.assert_allclose(toy_dataset_df.iloc[:, 0], toy_dataset_df.iloc[:, 1])
    np.testing.assert_allclose(toy_dataset_df.iloc[:, 2], toy_dataset_df.iloc[:, 3])
