from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from etna.datasets import TSDataset
from etna.reconciliation.base import BaseReconciliator


class DummyReconciliator(BaseReconciliator):
    def fit(self, ts: TSDataset) -> "DummyReconciliator":
        self.mapping_matrix = Mock()
        return self


@pytest.fixture
def hierarchical_ts():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "market": ["X"] * 2 + ["X"] * 2 + ["Y"] * 2 + ["Y"] * 2,
            "product": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1.0, 2.0] + [3.0, 4.0] + [5.0, 10.0] + [15.0, 20.0],
        }
    )
    df_exog = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02", "2000-01-03"] * 2,
            "market": ["X"] * 3 + ["Y"] * 3,
            "exog": [10.0, 12.0, 13.0] + [14.0, 15.0, 16.0],
        }
    )
    df, hs = TSDataset.to_hierarchical_dataset(
        df=df, level_columns=["market", "product"], keep_level_columns=False, return_hierarchy=True
    )
    df_exog, _ = TSDataset.to_hierarchical_dataset(
        df=df_exog, level_columns=["market"], keep_level_columns=False, return_hierarchy=False
    )
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, known_future=["exog"], hierarchical_structure=hs)
    return ts


@pytest.fixture
def hierarchical_ts_with_target_components(hierarchical_ts):
    target_components_df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "segment": ["X_a"] * 2 + ["X_b"] * 2 + ["Y_c"] * 2 + ["Y_d"] * 2,
            "target_component_a": [0.3, 0.27] + [0.7, 1.73] + [2, 2] + [3, 8],
            "target_component_b": [0.7, 1.73] + [2.3, 2.27] + [3, 8] + [12, 12],
        }
    )
    target_components_df = TSDataset.to_dataset(target_components_df)
    hierarchical_ts.add_target_components(target_components_df=target_components_df)
    return hierarchical_ts


@pytest.fixture
def market_total_mapping_matrix():
    mapping_matrix = np.array([[1, 1]])
    mapping_matrix = csr_matrix(mapping_matrix)
    return mapping_matrix


@pytest.fixture
def total_market_mapping_matrix():
    mapping_matrix = np.array([[1 / 6], [5 / 6]])
    mapping_matrix = csr_matrix(mapping_matrix)
    return mapping_matrix


@pytest.mark.parametrize("source_level", ("product", "market", "total"))
def test_aggregate(hierarchical_ts, source_level):
    reconciliator = DummyReconciliator(target_level="level", source_level=source_level)
    ts_aggregated = reconciliator.aggregate(ts=hierarchical_ts)
    assert ts_aggregated.current_df_level == source_level


def test_aggregate_fails_low_source_level(hierarchical_ts):
    ts_market_level = hierarchical_ts.get_level_dataset(target_level="market")
    reconciliator = DummyReconciliator(target_level="level", source_level="product")
    with pytest.raises(
        ValueError, match="Target level should be higher in the hierarchy than the current level of dataframe!"
    ):
        _ = reconciliator.aggregate(ts=ts_market_level)


def test_reconcile_not_fitted_fails(hierarchical_ts):
    reconciliator = DummyReconciliator(target_level="level", source_level="product")
    with pytest.raises(ValueError, match="Reconciliator is not fitted!"):
        _ = reconciliator.reconcile(ts=hierarchical_ts)


@pytest.mark.parametrize("cur_level", ("total", "product"))
def test_reconcile_wrong_level_fails(hierarchical_ts, cur_level, source_level="market"):
    hierarchical_ts = hierarchical_ts.get_level_dataset(target_level=cur_level)
    reconciliator = DummyReconciliator(target_level="level", source_level=source_level)
    reconciliator.fit(hierarchical_ts)
    with pytest.raises(ValueError, match=f"Dataset should be on the {source_level} level!"):
        _ = reconciliator.reconcile(ts=hierarchical_ts)


@pytest.mark.parametrize(
    "source_level, target_level, mapping_matrix",
    [("market", "total", "market_total_mapping_matrix"), ("total", "market", "total_market_mapping_matrix")],
)
def test_reconcile(hierarchical_ts, source_level, target_level, mapping_matrix, request):
    source_ts = hierarchical_ts.get_level_dataset(target_level=source_level)
    expected_ts = hierarchical_ts.get_level_dataset(target_level=target_level)

    reconciliator = DummyReconciliator(target_level=target_level, source_level=source_level)
    reconciliator.mapping_matrix = request.getfixturevalue(mapping_matrix)
    obtained_ts = reconciliator.reconcile(ts=source_ts)

    assert obtained_ts.freq == expected_ts.freq
    assert obtained_ts.current_df_level == expected_ts.current_df_level
    assert obtained_ts.known_future == expected_ts.known_future
    assert obtained_ts.regressors == expected_ts.regressors
    pd.testing.assert_frame_equal(obtained_ts.df, expected_ts.df)
    pd.testing.assert_frame_equal(obtained_ts.df_exog, expected_ts.df_exog)


@pytest.mark.parametrize(
    "source_level, target_level, mapping_matrix",
    [("market", "total", "market_total_mapping_matrix"), ("total", "market", "total_market_mapping_matrix")],
)
def test_reconcile_with_target_components(
    hierarchical_ts_with_target_components, source_level, target_level, mapping_matrix, request
):
    source_ts = hierarchical_ts_with_target_components.get_level_dataset(target_level=source_level)
    expected_ts = hierarchical_ts_with_target_components.get_level_dataset(target_level=target_level)

    reconciliator = DummyReconciliator(target_level=target_level, source_level=source_level)
    reconciliator.mapping_matrix = request.getfixturevalue(mapping_matrix)
    obtained_ts = reconciliator.reconcile(ts=source_ts)

    assert obtained_ts.freq == expected_ts.freq
    assert obtained_ts.current_df_level == expected_ts.current_df_level
    assert obtained_ts.known_future == expected_ts.known_future
    assert obtained_ts.regressors == expected_ts.regressors
    pd.testing.assert_frame_equal(obtained_ts.get_target_components(), expected_ts.get_target_components())
    pd.testing.assert_frame_equal(obtained_ts.df, expected_ts.df)
    pd.testing.assert_frame_equal(obtained_ts.df_exog, expected_ts.df_exog)
