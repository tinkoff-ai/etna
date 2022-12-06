from copy import deepcopy

import numpy as np
import pytest
from ruptures import Binseg

from etna.transforms import ChangePointsLevelTransform
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import STLTransform
from etna.transforms import TheilSenTrendTransform
from etna.transforms.decomposition import RupturesChangePointsModel


@pytest.mark.parametrize(
    "transform",
    (
        ChangePointsTrendTransform(
            in_column="target",
            change_points_model=RupturesChangePointsModel(
                change_points_model=Binseg(model="l2", jump=1, min_size=1),
                n_bkps=1,
            ),
        ),
        ChangePointsLevelTransform(
            in_column="target",
            change_points_model=RupturesChangePointsModel(
                change_points_model=Binseg(model="l2", jump=1, min_size=1),
                n_bkps=1,
            ),
        ),
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
