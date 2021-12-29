from copy import deepcopy
from typing import Any

import numpy as np
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
    example_tsds.fit_transform([bs])
    for segment in example_tsds.segments:
        assert abs(example_tsds[:, segment, "target"].mean()) < 1


@pytest.mark.parametrize(
    "custom_cost_class", (CostMl, CostAR, CostLinear, CostRbf, CostL2, CostL1, CostNormal, CostRank)
)
def test_binseg_run_with_custom_costs(example_tsds: TSDataset, custom_cost_class: Any):
    """Check that binseg trend works with different custom costs."""
    bs = BinsegTrendTransform(in_column="target", custom_cost=custom_cost_class())
    ts = deepcopy(example_tsds)
    ts.fit_transform([bs])
    ts.inverse_transform()
    assert (ts.df == example_tsds.df).all().all()


@pytest.mark.parametrize("model", ("l1", "l2", "normal", "rbf", "linear", "ar", "mahalanobis", "rank"))
def test_binseg_run_with_model(example_tsds: TSDataset, model: Any):
    """Check that binseg trend works with different models."""
    bs = BinsegTrendTransform(in_column="target", model=model)
    ts = deepcopy(example_tsds)
    ts.fit_transform([bs])
    ts.inverse_transform()
    assert (ts.df == example_tsds.df).all().all()


def test_binseg_runs_with_different_series_length(ts_with_different_series_length: TSDataset):
    """Check that binseg works with datasets with different length series."""
    bs = BinsegTrendTransform(in_column="target")
    ts = deepcopy(ts_with_different_series_length)
    ts.fit_transform([bs])
    ts.inverse_transform()
    np.allclose(ts.df.values, ts_with_different_series_length.df.values, equal_nan=True)


@pytest.mark.xfail
def test_fit_transform_with_nans(ts_diff_endings):
    transform = BinsegTrendTransform(in_column="target")
    ts_diff_endings.fit_transform([transform])
