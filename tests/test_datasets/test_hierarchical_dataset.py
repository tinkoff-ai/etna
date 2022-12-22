import pandas as pd
import pytest

from etna.datasets.hierarchical_structure import HierarchicalStructure
from etna.datasets.tsdataset import TSDataset


@pytest.fixture
def hierarchical_structure():
    hs = HierarchicalStructure(
        level_structure={"total": ["X", "Y"], "X": ["a", "b"], "Y": ["c", "d"]},
        level_names=["total", "market", "product"],
    )
    return hs


@pytest.fixture
def different_level_segments_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["a"] * 2,
            "target": [1, 2] + [10, 20],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def different_level_segments_df_exog():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["a"] * 2,
            "exog": [1, 2] + [10, 20],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def missing_segments_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"],
            "segment": ["X"] * 2,
            "target": [1, 2],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [1, 2] + [10, 20],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_df_exog():
    df_exog = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "exog": [1, 2] + [10, 20],
        }
    )
    df_exog = TSDataset.to_dataset(df_exog)
    return df_exog


@pytest.fixture
def product_level_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "segment": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1, 2] + [10, 20] + [100, 200] + [1000, 2000],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def simple_hierarchical_ts(market_level_df, hierarchical_structure):
    df = market_level_df
    ts = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)
    return ts


def test_get_dataframe_level_different_level_segments_fails(different_level_segments_df, simple_hierarchical_ts):
    with pytest.raises(ValueError, match="Segments in dataframe are from more than 1 hierarchical levels!"):
        simple_hierarchical_ts._get_dataframe_level(df=different_level_segments_df)


def test_get_dataframe_level_missing_segments_fails(missing_segments_df, simple_hierarchical_ts):
    with pytest.raises(ValueError, match="Some segments of hierarchical level are missing in dataframe!"):
        simple_hierarchical_ts._get_dataframe_level(df=missing_segments_df)


@pytest.mark.parametrize("df, expected_level", [("market_level_df", "market"), ("product_level_df", "product")])
def test_get_dataframe(df, expected_level, simple_hierarchical_ts, request):
    df = request.getfixturevalue(df)
    df_level = simple_hierarchical_ts._get_dataframe_level(df=df)
    assert df_level == expected_level


def test_init_different_level_segments_df_fails(different_level_segments_df, hierarchical_structure):
    df = different_level_segments_df
    with pytest.raises(ValueError, match="Segments in dataframe are from more than 1 hierarchical levels!"):
        _ = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)


def test_init_different_level_segments_df_exog_fails(market_level_df, different_level_segments_df_exog, hierarchical_structure):
    df, df_exog = market_level_df, different_level_segments_df_exog
    with pytest.raises(ValueError, match="Segments in dataframe are from more than 1 hierarchical levels!"):
        _ = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)


def test_init_df_same_level_df_exog(market_level_df, market_level_df_exog, hierarchical_structure):
    df, df_exog = market_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    df_columns = set(ts.columns.get_level_values("feature"))
    assert df_columns == {"target", "exog"}


def test_init_df_different_level_df_exog(product_level_df, market_level_df_exog, hierarchical_structure):
    df, df_exog = product_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    df_columns = set(ts.columns.get_level_values("feature"))
    assert df_columns == {"target"}


def test_make_future_df_same_level_df_exog(market_level_df, market_level_df_exog, hierarchical_structure):
    df, df_exog = market_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    future = ts.make_future(future_steps=4)
    future_columns = set(future.columns.get_level_values("feature"))
    assert future_columns == {"target", "exog"}


def test_make_future_df_different_level_df_exog(product_level_df, market_level_df_exog, hierarchical_structure):
    df, df_exog = product_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    future = ts.make_future(future_steps=4)
    future_columns = set(future.columns.get_level_values("feature"))
    assert future_columns == {"target"}


def test_level_names_with_hierarchical_structure(simple_hierarchical_ts, expected_names=["total", "market", "product"]):
    ts_level_names = simple_hierarchical_ts.level_names()
    assert sorted(ts_level_names) == sorted(expected_names)


def test_level_names_without_hierarchical_structure(market_level_df):
    ts = TSDataset(df=market_level_df, freq="D")
    ts_level_names = ts.level_names()
    assert ts_level_names is None
