import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.reconciliation.base import BaseReconciliator


@pytest.fixture
def hierarchical_ts():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "market": ["X"] * 2 + ["X"] * 2 + ["Y"] * 2 + ["Y"] * 2,
            "product": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1, 2] + [10, 20] + [100, 200] + [1000, 2000],
        }
    )
    df, hs = TSDataset.to_hierarchical_dataset(
        df=df, level_columns=["market", "product"], keep_level_columns=False, return_hierarchy=True
    )
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hs)
    return ts


@pytest.mark.parametrize("source_level", ("product", "market", "total"))
def test_aggregate(hierarchical_ts, source_level):
    reconciliator = BaseReconciliator(target_level="level", source_level=source_level)
    ts_aggregated = reconciliator.aggregate(ts=hierarchical_ts)
    assert ts_aggregated.current_df_level == source_level


def test_aggregate_fails_low_source_level(hierarchical_ts):
    ts_market_level = hierarchical_ts.get_level_dataset(target_level="market")
    reconciliator = BaseReconciliator(target_level="level", source_level="product")
    with pytest.raises(ValueError, match=""):
        _ = reconciliator.aggregate(ts=ts_market_level)
