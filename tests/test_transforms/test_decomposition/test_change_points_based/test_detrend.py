from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg
from ruptures.costs import CostAR
from ruptures.costs import CostL1
from ruptures.costs import CostL2
from ruptures.costs import CostLinear
from ruptures.costs import CostMl
from ruptures.costs import CostNormal
from ruptures.costs import CostRank
from ruptures.costs import CostRbf

from etna.datasets import TSDataset
from etna.transforms.decomposition import ChangePointsTrendTransform
from etna.transforms.decomposition import RupturesChangePointsModel
from etna.transforms.decomposition.change_points_based.detrend import _OneSegmentChangePointsTrendTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


def test_binseg_in_pipeline(example_tsds: TSDataset):
    bs = ChangePointsTrendTransform(in_column="target")
    bs.fit_transform(example_tsds)
    for segment in example_tsds.segments:
        assert abs(example_tsds[:, segment, "target"].mean()) < 1


@pytest.mark.parametrize(
    "custom_cost_class", (CostMl, CostAR, CostLinear, CostRbf, CostL2, CostL1, CostNormal, CostRank)
)
def test_binseg_run_with_custom_costs(example_tsds: TSDataset, custom_cost_class: Any):
    """Check that binseg trend works with different custom costs."""
    bs = ChangePointsTrendTransform(
        in_column="target",
        change_points_model=RupturesChangePointsModel(
            change_points_model=Binseg(custom_cost=custom_cost_class()),
            n_bkps=5,
        ),
    )
    ts = deepcopy(example_tsds)
    bs.fit_transform(ts)
    bs.inverse_transform(ts)
    assert (ts.to_pandas() == example_tsds.to_pandas()).all().all()


@pytest.mark.parametrize("model", ("l1", "l2", "normal", "rbf", "linear", "ar", "mahalanobis", "rank"))
def test_binseg_run_with_model(example_tsds: TSDataset, model: Any):
    """Check that binseg trend works with different models."""
    bs = ChangePointsTrendTransform(
        in_column="target",
        change_points_model=RupturesChangePointsModel(
            change_points_model=Binseg(model=model),
            n_bkps=5,
        ),
    )
    ts = deepcopy(example_tsds)
    bs.fit_transform(ts)
    bs.inverse_transform(ts)
    assert (ts.to_pandas() == example_tsds.to_pandas()).all().all()


def test_binseg_runs_with_different_series_length(ts_with_different_series_length: TSDataset):
    """Check that binseg works with datasets with different length series."""
    bs = ChangePointsTrendTransform(in_column="target")
    ts = deepcopy(ts_with_different_series_length)
    bs.fit_transform(ts)
    bs.inverse_transform(ts)
    np.allclose(ts.to_pandas().values, ts_with_different_series_length.to_pandas().values, equal_nan=True)


def test_fit_transform_with_nans_in_tails(ts_with_nans_in_tails):
    transform = ChangePointsTrendTransform(in_column="target")
    transformed_df = transform.fit_transform(ts=ts_with_nans_in_tails).to_pandas()
    for segment in transformed_df.columns.get_level_values("segment").unique():
        segment_slice = transformed_df.loc[pd.IndexSlice[:], pd.IndexSlice[segment, :]][segment]
        assert abs(segment_slice["target"].mean()) < 0.1


def test_fit_transform_with_nans_in_middle_raise_error(ts_with_nans):
    transform = ChangePointsTrendTransform(in_column="target")
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        transform.fit_transform(ts=ts_with_nans)


def test_get_features(example_tsds: TSDataset):
    """Check that _get_features method works correctly."""
    segment_df = example_tsds[:, "segment_1", :]
    features = _OneSegmentChangePointsTrendTransform._get_features(series=segment_df)
    assert isinstance(features, np.ndarray)
    assert features.shape == (len(segment_df), 1)
    assert isinstance(features[0][0], float)


def test_save_load(example_tsds):
    transform = ChangePointsTrendTransform(in_column="target")
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)


@pytest.mark.parametrize(
    "transform, expected_length",
    [
        (ChangePointsTrendTransform(in_column="target"), 2),
        (
            ChangePointsTrendTransform(
                in_column="target",
                change_points_model=RupturesChangePointsModel(
                    change_points_model=Binseg(model="ar"),
                    n_bkps=5,
                ),
            ),
            2,
        ),
        (
            ChangePointsTrendTransform(
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
def test_params_to_tune(transform, expected_length, example_tsds):
    ts = example_tsds
    assert len(transform.params_to_tune()) == expected_length
    assert_sampling_is_valid(transform=transform, ts=ts)
