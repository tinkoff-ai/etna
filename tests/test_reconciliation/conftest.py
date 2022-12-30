import pandas as pd
import pytest

from etna.datasets import TSDataset


@pytest.fixture
def market_level_df_w_negatives():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [-1.0, 2.0] + [0, -20.0],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def total_level_simple_hierarchical_ts(total_level_df, hierarchical_structure):
    ts = TSDataset(df=total_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_simple_hierarchical_ts(market_level_df, hierarchical_structure):
    ts = TSDataset(df=market_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def product_level_simple_hierarchical_ts(product_level_df, hierarchical_structure):
    ts = TSDataset(df=product_level_df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def market_level_simple_hierarchical_ts_w_nans(market_level_df_w_nans, hierarchical_structure):
    ts = TSDataset(df=market_level_df_w_nans, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def simple_hierarchical_ts_w_negatives(market_level_df_w_negatives, hierarchical_structure):
    ts = TSDataset(df=market_level_df_w_negatives, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


@pytest.fixture
def simple_no_hierarchy_ts(market_level_df):
    ts = TSDataset(df=market_level_df, freq="D")
    return ts
