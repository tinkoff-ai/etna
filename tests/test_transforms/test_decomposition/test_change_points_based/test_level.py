from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.transforms import ChangePointsLevelTransform
from etna.transforms.decomposition.change_points_based.change_points_models import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_based.per_interval_models import MeanPerIntervalModel
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.fixture
def ts_with_local_levels(random_seed) -> TSDataset:
    df_1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100)})
    df_1["segment"] = "segment_1"
    df_1["target"] = np.concatenate(
        [
            np.random.normal(size=(20,), scale=1, loc=10),
            np.random.normal(size=(40,), scale=1, loc=100),
            np.random.normal(size=(30,), scale=1, loc=-1),
            np.random.normal(size=(10,), scale=1, loc=-20),
        ]
    )
    df_2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100)})
    df_2["segment"] = "segment_2"
    df_2["target"] = np.concatenate(
        [
            np.random.normal(size=(30,), scale=1, loc=15),
            np.random.normal(size=(30,), scale=1, loc=0),
            np.random.normal(size=(20,), scale=1, loc=-11),
            np.random.normal(size=(20,), scale=1, loc=12),
        ]
    )
    df = pd.concat((df_1, df_2))
    df = TSDataset.to_dataset(df=df)
    ts = TSDataset(df, "D")
    return ts


def test_level_transform(ts_with_local_levels: TSDataset):
    transform = ChangePointsLevelTransform(
        in_column="target",
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="l2"), n_bkps=4),
        per_interval_model=MeanPerIntervalModel(),
    )
    ts_with_local_levels.fit_transform(transforms=[transform])
    np.allclose(ts_with_local_levels.df.mean().values, 0)
    assert np.all(x < 3 for x in ts_with_local_levels.df.std().values)


def test_level_transform_inverse_transform(ts_with_local_levels: TSDataset):
    original_ts = deepcopy(ts_with_local_levels)
    transform = ChangePointsLevelTransform(
        in_column="target",
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="l2"), n_bkps=4),
        per_interval_model=MeanPerIntervalModel(),
    )
    ts_with_local_levels.fit_transform(transforms=[transform])
    ts_with_local_levels.inverse_transform(transforms=[transform])
    np.testing.assert_array_almost_equal(ts_with_local_levels.df, original_ts.df)


def test_save_load(ts_with_local_levels):
    transform = ChangePointsLevelTransform(
        in_column="target",
        change_points_model=RupturesChangePointsModel(change_points_model=Binseg(model="l2"), n_bkps=4),
        per_interval_model=MeanPerIntervalModel(),
    )
    assert_transformation_equals_loaded_original(transform=transform, ts=ts_with_local_levels)


@pytest.mark.parametrize(
    "transform, expected_length",
    [
        (ChangePointsLevelTransform(in_column="target"), 2),
        (
            ChangePointsLevelTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="ar"),
                    n_bkps=5,
                ),
            ),
            2,
        ),
        (
            ChangePointsLevelTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="ar"),
                    n_bkps=10,
                ),
            ),
            0,
        ),
    ],
)
def test_params_to_tune(transform, expected_length, ts_with_local_levels):
    ts = ts_with_local_levels
    assert len(transform.params_to_tune()) == expected_length
    assert_sampling_is_valid(transform=transform, ts=ts)
