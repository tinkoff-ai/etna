import pandas as pd
import pytest

from etna.datasets.hierarchical_structure import HierarchicalStructure
from etna.datasets.tsdataset import TSDataset


@pytest.fixture
def hierarchical_structure_complex():
    hs = HierarchicalStructure(
        level_structure={
            "total": ["77", "120"],
            "77": ["77_X"],
            "120": ["120_Y"],
            "77_X": ["77_X_1", "77_X_2"],
            "120_Y": ["120_Y_3", "120_Y_4"],
        },
        level_names=["total", "categorical", "string", "int"],
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
def level_columns_different_types_df():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "categorical": [77] * 4 + [120] * 4,
            "string": ["X"] * 2 + ["X"] * 2 + ["Y"] * 2 + ["Y"] * 2,
            "int": [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2,
            "target": [1, 2] + [10, 20] + [100, 200] + [1000, 2000],
        }
    )
    df["categorical"] = df["categorical"].astype("category")
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
def product_level_df_long():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "market": ["X"] * 2 + ["X"] * 2 + ["Y"] * 2 + ["Y"] * 2,
            "product": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [1, 2] + [10, 20] + [100, 200] + [1000, 2000],
        }
    )
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
def product_level_df_wide(product_level_df_long):
    df = product_level_df_long
    df["segment"] = ["X_a"] * 2 + ["X_b"] * 2 + ["Y_c"] * 2 + ["Y_d"] * 2
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def market_level_df_exog():
    df_exog = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-03"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "exog": [1.0, 5.0] + [10.0, 5.0],
            "regressor": 1,
        }
    )
    df_exog = TSDataset.to_dataset(df_exog)
    return df_exog


@pytest.fixture
def l4_level_df_long():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["c"] * 2 + ["d"] * 2,
            "target": [0, 1] + [2, 3],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l3_level_df_long():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["a"] * 2 + ["b"] * 2,
            "target": [0, 1] + [2, 3],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l2_level_df_long():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [0, 1] + [2, 3],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l1_level_df_long():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"],
            "segment": ["total"] * 2,
            "target": [2, 4],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l4_level_df_tailed():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 4,
            "segment": ["e"] * 2 + ["h"] * 2 + ["f"] * 2 + ["g"] * 2,
            "target": [0, 1] + [2, 3] + [4, 5] + [6, 7],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l3_level_df_tailed():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 3,
            "segment": ["a"] * 2 + ["c"] * 2 + ["d"] * 2,
            "target": [2, 4] + [4, 5] + [6, 7],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l2_level_df_tailed():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"] * 2,
            "segment": ["X"] * 2 + ["Y"] * 2,
            "target": [2, 4] + [10, 12],
        }
    )
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def l1_level_df_tailed():
    df = pd.DataFrame(
        {
            "timestamp": ["2000-01-01", "2000-01-02"],
            "segment": ["total"] * 2,
            "target": [12, 16],
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


def test_init_different_level_segments_df_exog_fails(
    market_level_df, different_level_segments_df_exog, hierarchical_structure
):
    df, df_exog = market_level_df, different_level_segments_df_exog
    with pytest.raises(ValueError, match="Segments in dataframe are from more than 1 hierarchical levels!"):
        _ = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)


def test_init_df_same_level_df_exog(
    market_level_df, market_level_df_exog, hierarchical_structure, expected_columns={"target", "regressor", "exog"}
):
    df, df_exog = market_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    df_columns = set(ts.columns.get_level_values("feature"))
    assert df_columns == expected_columns


def test_init_df_different_level_df_exog(
    product_level_df, market_level_df_exog, hierarchical_structure, expected_columns={"target"}
):
    df, df_exog = product_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    df_columns = set(ts.columns.get_level_values("feature"))
    assert df_columns == expected_columns


def test_init_missing_segmnets_df(missing_segments_df, hierarchical_structure):
    df = missing_segments_df
    with pytest.raises(ValueError, match="Some segments of hierarchical level are missing in dataframe!"):
        _ = TSDataset(df=df, freq="D", hierarchical_structure=hierarchical_structure)


def test_make_future_df_same_level_df_exog(
    market_level_df, market_level_df_exog, hierarchical_structure, expected_columns={"target", "regressor", "exog"}
):
    df, df_exog = market_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    future = ts.make_future(future_steps=4)
    future_columns = set(future.columns.get_level_values("feature"))
    assert future_columns == expected_columns


def test_make_future_df_different_level_df_exog(
    product_level_df, market_level_df_exog, hierarchical_structure, expected_columns={"target"}
):
    df, df_exog = product_level_df, market_level_df_exog
    ts = TSDataset(df=df, freq="D", df_exog=df_exog, hierarchical_structure=hierarchical_structure)
    future = ts.make_future(future_steps=4)
    future_columns = set(future.columns.get_level_values("feature"))
    assert future_columns == expected_columns


def test_level_names_with_hierarchical_structure(simple_hierarchical_ts, expected_names=["total", "market", "product"]):
    ts_level_names = simple_hierarchical_ts.level_names()
    assert sorted(ts_level_names) == sorted(expected_names)


def test_level_names_without_hierarchical_structure(market_level_df):
    df = market_level_df
    ts = TSDataset(df=df, freq="D")
    ts_level_names = ts.level_names()
    assert ts_level_names is None


def test_to_hierarchical_dataset_fails_empty_level_columns(product_level_df_long):
    df = product_level_df_long
    with pytest.raises(ValueError, match="Value of level_columns shouldn't be empty"):
        _ = TSDataset.to_hierarchical_dataset(df=df, level_columns=[])


def test_to_hierarchical_dataset_not_change_input_df(product_level_df_long):
    df = product_level_df_long
    df_before = df.copy()
    df_after, _ = TSDataset.to_hierarchical_dataset(
        df=df, level_columns=["market", "product"], keep_level_columns=False, return_hierarchy=True
    )
    pd.testing.assert_frame_equal(df, df_before)


@pytest.mark.parametrize(
    "df_fixture, level_columns, sep, expected_segments",
    [
        ("product_level_df_long", ["market", "product"], "_", ["X_a", "X_b", "Y_c", "Y_d"]),
        ("product_level_df_long", ["market", "product"], "#", ["X#a", "X#b", "Y#c", "Y#d"]),
        ("product_level_df_long", ["product"], "_", ["a", "b", "c", "d"]),
        (
            "level_columns_different_types_df",
            ["categorical", "string", "int"],
            "_",
            ["77_X_1", "77_X_2", "120_Y_3", "120_Y_4"],
        ),
    ],
)
def test_to_hierarchical_dataset_correct_segments(df_fixture, level_columns, sep, expected_segments, request):
    df = request.getfixturevalue(df_fixture)
    df, _ = TSDataset.to_hierarchical_dataset(df=df, level_columns=level_columns, sep=sep, return_hierarchy=True)
    df_segments = df.columns.get_level_values("segment").unique()
    assert sorted(df_segments) == sorted(expected_segments)


@pytest.mark.parametrize(
    "keep_level_columns, expected_columns", [(True, ["target", "market", "product"]), (False, ["target"])]
)
def test_to_hierarchical_dataset_correct_columns(product_level_df_long, keep_level_columns, expected_columns):
    df = product_level_df_long
    df, _ = TSDataset.to_hierarchical_dataset(
        df=df, keep_level_columns=keep_level_columns, level_columns=["market", "product"], return_hierarchy=True
    )
    df_columns = df.columns.get_level_values("feature").unique()
    assert sorted(df_columns) == sorted(expected_columns)


def test_to_hierarchical_dataset_correct_dataframe(product_level_df_long, product_level_df_wide):
    df_wide_obtained, _ = TSDataset.to_hierarchical_dataset(
        df=product_level_df_long, keep_level_columns=True, level_columns=["market", "product"], return_hierarchy=True
    )
    pd.testing.assert_frame_equal(df_wide_obtained, product_level_df_wide)


def test_to_hierarchical_dataset_hierarchical_structure(
    level_columns_different_types_df, hierarchical_structure_complex
):
    _, hs = TSDataset.to_hierarchical_dataset(
        df=level_columns_different_types_df, level_columns=["categorical", "string", "int"], return_hierarchy=True
    )
    assert hs.level_names == hierarchical_structure_complex.level_names
    for level_name in hierarchical_structure_complex.level_names:
        assert level_name in hs.level_names
        expected_level_segments = hierarchical_structure_complex.get_level_segments(level_name=level_name)
        obtained_level_segments = hs.get_level_segments(level_name=level_name)
        assert sorted(obtained_level_segments) == sorted(expected_level_segments)


@pytest.mark.parametrize(
    "hierarchical_structure_name,source_df_name,target_level,target_df_name",
    (
        ("hierarchical_structure", "product_level_df", "market", "market_level_df"),
        ("hierarchical_structure", "product_level_df", "total", "total_level_df"),
        ("hierarchical_structure", "market_level_df", "total", "total_level_df"),
        ("hierarchical_structure", "product_level_df_w_nans", "market", "market_level_df_w_nans"),
        ("hierarchical_structure", "product_level_df_w_nans", "total", "total_level_df_w_nans"),
        ("hierarchical_structure", "market_level_df_w_nans", "total", "total_level_df_w_nans"),
        ("long_hierarchical_structure", "l4_level_df_long", "l3", "l3_level_df_long"),
        ("long_hierarchical_structure", "l4_level_df_long", "l2", "l2_level_df_long"),
        ("long_hierarchical_structure", "l4_level_df_long", "l1", "l1_level_df_long"),
        ("long_hierarchical_structure", "l3_level_df_long", "l2", "l2_level_df_long"),
        ("long_hierarchical_structure", "l3_level_df_long", "l1", "l1_level_df_long"),
        ("long_hierarchical_structure", "l2_level_df_long", "l1", "l1_level_df_long"),
        ("tailed_hierarchical_structure", "l4_level_df_tailed", "l3", "l3_level_df_tailed"),
        ("tailed_hierarchical_structure", "l4_level_df_tailed", "l2", "l2_level_df_tailed"),
        ("tailed_hierarchical_structure", "l4_level_df_tailed", "l1", "l1_level_df_tailed"),
        ("tailed_hierarchical_structure", "l3_level_df_tailed", "l2", "l2_level_df_tailed"),
        ("tailed_hierarchical_structure", "l3_level_df_tailed", "l1", "l1_level_df_tailed"),
        ("tailed_hierarchical_structure", "l2_level_df_tailed", "l1", "l1_level_df_tailed"),
    ),
)
def test_get_level_dataset(hierarchical_structure_name, source_df_name, target_level, target_df_name, request):
    hierarchical_structure = request.getfixturevalue(hierarchical_structure_name)

    source_df = request.getfixturevalue(source_df_name)
    source_ts = TSDataset(df=source_df, freq="D", hierarchical_structure=hierarchical_structure)

    target_df = request.getfixturevalue(target_df_name)
    target_ts = TSDataset(df=target_df, freq="D", hierarchical_structure=hierarchical_structure)

    estimated_target_ts = source_ts.get_level_dataset(target_level)

    # check attributes
    assert target_ts.freq == estimated_target_ts.freq
    assert target_ts.hierarchical_structure == estimated_target_ts.hierarchical_structure
    assert target_ts.current_df_level == estimated_target_ts.current_df_level

    pd.testing.assert_frame_equal(target_ts.df, estimated_target_ts.df)


@pytest.mark.parametrize(
    "source_df_name,target_level,target_df_name",
    (
        ("product_level_df", "market", "market_level_df"),
        ("product_level_df", "total", "total_level_df"),
        ("market_level_df", "total", "total_level_df"),
    ),
)
def test_get_level_dataset_with_exog(
    source_df_name, target_level, target_df_name, market_level_df_exog, hierarchical_structure, request
):
    source_df = request.getfixturevalue(source_df_name)
    source_ts = TSDataset(
        df=source_df,
        df_exog=market_level_df_exog,
        freq="D",
        hierarchical_structure=hierarchical_structure,
        known_future=["regressor"],
    )

    target_df = request.getfixturevalue(target_df_name)
    target_ts = TSDataset(
        df=target_df,
        df_exog=market_level_df_exog,
        freq="D",
        hierarchical_structure=hierarchical_structure,
        known_future=["regressor"],
    )

    estimated_target_ts = source_ts.get_level_dataset(target_level)

    assert target_ts.current_df_exog_level == estimated_target_ts.current_df_exog_level
    pd.testing.assert_frame_equal(target_ts.df, estimated_target_ts.df)


def test_get_level_dataset_no_hierarchy_error(market_level_df):
    ts = TSDataset(df=market_level_df, freq="D")
    with pytest.raises(ValueError, match="Method could be applied only to instances with a hierarchy!"):
        ts.get_level_dataset(target_level="total")


@pytest.mark.parametrize(
    "target_level",
    ("", "abcd"),
)
def test_get_level_dataset_invalid_level_name_error(simple_hierarchical_ts, target_level):
    with pytest.raises(ValueError, match=f"Invalid level name: {target_level}"):
        simple_hierarchical_ts.get_level_dataset(target_level=target_level)


def test_get_level_dataset_lower_level_error(simple_hierarchical_ts):
    with pytest.raises(
        ValueError, match="Target level should be higher in the hierarchy than the current level of dataframe!"
    ):
        simple_hierarchical_ts.get_level_dataset(target_level="product")


@pytest.mark.parametrize(
    "target_level, expected_dataframe_name",
    (
        ("product", "product_level_constant_forecast_with_quantiles"),
        ("market", "market_level_constant_forecast_with_quantiles"),
        ("total", "total_level_constant_forecast_with_quantiles"),
    ),
)
def test_get_level_dataset_with_quantiles(
    product_level_constant_forecast_with_quantiles, target_level, expected_dataframe_name, request
):
    expected_df = request.getfixturevalue(expected_dataframe_name).to_pandas()
    reconciled_df = product_level_constant_forecast_with_quantiles.get_level_dataset(
        target_level=target_level
    ).to_pandas()
    pd.testing.assert_frame_equal(reconciled_df, expected_df)


@pytest.mark.parametrize(
    "target_level, expected_dataframe_name",
    (
        ("product", "product_level_constant_forecast_with_target_components"),
        ("market", "market_level_constant_forecast_with_target_components"),
        ("total", "total_level_constant_forecast_with_target_components"),
    ),
)
def test_get_level_dataset_with_target_components(
    product_level_constant_forecast_with_target_components, target_level, expected_dataframe_name, request
):
    expected_ts = request.getfixturevalue(expected_dataframe_name)
    reconciled_ts = product_level_constant_forecast_with_target_components.get_level_dataset(target_level=target_level)
    pd.testing.assert_frame_equal(reconciled_ts.get_target_components(), expected_ts.get_target_components())
    pd.testing.assert_frame_equal(reconciled_ts.to_pandas(), expected_ts.to_pandas())
