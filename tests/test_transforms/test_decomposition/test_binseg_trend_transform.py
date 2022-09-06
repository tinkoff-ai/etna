from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ruptures.costs import CostAR
from ruptures.costs import CostL1
from ruptures.costs import CostL2
from ruptures.costs import CostLinear
from ruptures.costs import CostMl
from ruptures.costs import CostNormal
from ruptures.costs import CostRank
from ruptures.costs import CostRbf

from etna.datasets import TSDataset
from etna.transforms.decomposition import BinsegTrendTransform


def test_binseg_in_pipeline(example_tsds: TSDataset):
    bs = BinsegTrendTransform(in_column="target")
    bs.fit_transform(example_tsds)
    for segment in example_tsds.segments:
        assert abs(example_tsds[:, segment, "target"].mean()) < 1


@pytest.mark.parametrize(
    "custom_cost_class", (CostMl, CostAR, CostLinear, CostRbf, CostL2, CostL1, CostNormal, CostRank)
)
def test_binseg_run_with_custom_costs(example_tsds: TSDataset, custom_cost_class: Any):
    """Check that binseg trend works with different custom costs."""
    bs = BinsegTrendTransform(in_column="target", custom_cost=custom_cost_class())
    ts = deepcopy(example_tsds)
    bs.fit_transform(ts)
    bs.inverse_transform(ts)
    assert (ts.to_pandas() == example_tsds.to_pandas()).all().all()


@pytest.mark.parametrize("model", ("l1", "l2", "normal", "rbf", "linear", "ar", "mahalanobis", "rank"))
def test_binseg_run_with_model(example_tsds: TSDataset, model: Any):
    """Check that binseg trend works with different models."""
    bs = BinsegTrendTransform(in_column="target", model=model)
    ts = deepcopy(example_tsds)
    bs.fit_transform(ts)
    bs.inverse_transform(ts)
    assert (ts.to_pandas() == example_tsds.to_pandas()).all().all()


def test_binseg_runs_with_different_series_length(ts_with_different_series_length: TSDataset):
    """Check that binseg works with datasets with different length series."""
    bs = BinsegTrendTransform(in_column="target")
    ts = deepcopy(ts_with_different_series_length)
    bs.fit_transform(ts)
    bs.inverse_transform(ts)
    np.allclose(ts.to_pandas().values, ts_with_different_series_length.to_pandas().values, equal_nan=True)


def test_fit_transform_with_nans_in_tails(ts_with_nans_in_tails):
    transform = BinsegTrendTransform(in_column="target")
    transformed_df = transform.fit_transform(ts=ts_with_nans_in_tails).to_pandas()
    for segment in transformed_df.columns.get_level_values("segment").unique():
        segment_slice = transformed_df.loc[pd.IndexSlice[:], pd.IndexSlice[segment, :]][segment]
        assert abs(segment_slice["target"].mean()) < 0.1


def test_fit_transform_with_nans_in_middle_raise_error(ts_with_nans):
    transform = BinsegTrendTransform(in_column="target")
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        transform.fit_transform(ts=ts_with_nans)
