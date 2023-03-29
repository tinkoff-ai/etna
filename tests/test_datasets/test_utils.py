import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data
from etna.datasets import generate_ar_df
from etna.datasets.utils import _TorchDataset
from etna.datasets.utils import get_level_dataframe
from etna.datasets.utils import get_target_with_quantiles
from etna.datasets.utils import inverse_transform_target_components
from etna.datasets.utils import match_target_components
from etna.datasets.utils import set_columns_wide


@pytest.fixture
def df_exog_no_segments() -> pd.DataFrame:
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"timestamp": timestamp, "exog_1": 1, "exog_2": 2, "exog_3": 3})
    return df


def test_duplicate_data_fail_empty_segments(df_exog_no_segments):
    """Test that `duplicate_data` fails on empty list of segments."""
    with pytest.raises(ValueError, match="Parameter segments shouldn't be empty"):
        _ = duplicate_data(df=df_exog_no_segments, segments=[])


def test_duplicate_data_fail_wrong_format(df_exog_no_segments):
    """Test that `duplicate_data` fails on wrong given format."""
    with pytest.raises(ValueError, match="'wrong_format' is not a valid DataFrameFormat"):
        _ = duplicate_data(df=df_exog_no_segments, segments=["segment_1", "segment_2"], format="wrong_format")


def test_duplicate_data_fail_wrong_df(df_exog_no_segments):
    """Test that `duplicate_data` fails on wrong df."""
    with pytest.raises(ValueError, match="There should be 'timestamp' column"):
        _ = duplicate_data(df=df_exog_no_segments.drop(columns=["timestamp"]), segments=["segment_1", "segment_2"])


def test_duplicate_data_long_format(df_exog_no_segments):
    """Test that `duplicate_data` makes duplication in long format."""
    segments = ["segment_1", "segment_2"]
    df_duplicated = duplicate_data(df=df_exog_no_segments, segments=segments, format="long")
    expected_columns = set(df_exog_no_segments.columns)
    expected_columns.add("segment")
    assert set(df_duplicated.columns) == expected_columns
    for segment in segments:
        df_temp = df_duplicated[df_duplicated["segment"] == segment].reset_index(drop=True)
        for column in df_exog_no_segments.columns:
            assert np.all(df_temp[column] == df_exog_no_segments[column])


def test_duplicate_data_wide_format(df_exog_no_segments):
    """Test that `duplicate_data` makes duplication in wide format."""
    segments = ["segment_1", "segment_2"]
    df_duplicated = duplicate_data(df=df_exog_no_segments, segments=segments, format="wide")
    expected_columns_segment = set(df_exog_no_segments.columns)
    expected_columns_segment.remove("timestamp")
    for segment in segments:
        df_temp = df_duplicated.loc[:, pd.IndexSlice[segment, :]]
        df_temp.columns = df_temp.columns.droplevel("segment")
        assert set(df_temp.columns) == expected_columns_segment
        assert np.all(df_temp.index == df_exog_no_segments["timestamp"])
        for column in df_exog_no_segments.columns.drop("timestamp"):
            assert np.all(df_temp[column].values == df_exog_no_segments[column].values)


def test_torch_dataset():
    """Unit test for `_TorchDataset` class."""
    ts_samples = [{"decoder_target": np.array([1, 2, 3]), "encoder_target": np.array([1, 2, 3])}]

    torch_dataset = _TorchDataset(ts_samples=ts_samples)

    assert torch_dataset[0] == ts_samples[0]
    assert len(torch_dataset) == 1


def _get_df_wide(random_seed: int) -> pd.DataFrame:
    df = generate_ar_df(periods=5, start_time="2020-01-01", n_segments=3, random_seed=random_seed)
    df_wide = TSDataset.to_dataset(df)

    df_exog = df.copy()
    df_exog = df_exog.rename(columns={"target": "exog_0"})
    df_exog["exog_0"] = df_exog["exog_0"] + 1
    df_exog["exog_1"] = df_exog["exog_0"] + 1
    df_exog["exog_2"] = df_exog["exog_1"] + 1
    df_exog_wide = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq="D")
    df = ts.df

    # make some reorderings for checking corner cases
    df = df.loc[:, pd.IndexSlice[["segment_2", "segment_0", "segment_1"], ["target", "exog_2", "exog_1", "exog_0"]]]

    return df


@pytest.fixture
def df_left() -> pd.DataFrame:
    return _get_df_wide(0)


@pytest.fixture
def df_right() -> pd.DataFrame:
    return _get_df_wide(1)


@pytest.mark.parametrize(
    "features_left, features_right",
    [
        (None, None),
        (["exog_0"], ["exog_0"]),
        (["exog_0", "exog_1"], ["exog_0", "exog_1"]),
        (["exog_0", "exog_1"], ["exog_1", "exog_2"]),
    ],
)
@pytest.mark.parametrize(
    "segments_left, segment_right",
    [
        (None, None),
        (["segment_0"], ["segment_0"]),
        (["segment_0", "segment_1"], ["segment_0", "segment_1"]),
        (["segment_0", "segment_1"], ["segment_1", "segment_2"]),
    ],
)
@pytest.mark.parametrize(
    "timestamps_idx_left, timestamps_idx_right", [(None, None), ([0], [0]), ([1, 2], [1, 2]), ([1, 2], [3, 4])]
)
def test_set_columns_wide(
    timestamps_idx_left,
    timestamps_idx_right,
    segments_left,
    segment_right,
    features_left,
    features_right,
    df_left,
    df_right,
):
    timestamps_left = None if timestamps_idx_left is None else df_left.index[timestamps_idx_left]
    timestamps_right = None if timestamps_idx_right is None else df_right.index[timestamps_idx_right]

    df_obtained = set_columns_wide(
        df_left,
        df_right,
        timestamps_left=timestamps_left,
        timestamps_right=timestamps_right,
        segments_left=segments_left,
        segments_right=segment_right,
        features_left=features_left,
        features_right=features_right,
    )

    # get expected result
    df_expected = df_left.copy()

    timestamps_left_full = df_left.index.tolist() if timestamps_left is None else timestamps_left
    timestamps_right_full = df_right.index.tolist() if timestamps_left is None else timestamps_right

    segments_left_full = (
        df_left.columns.get_level_values("segment").unique().tolist() if segments_left is None else segments_left
    )
    segments_right_full = (
        df_left.columns.get_level_values("segment").unique().tolist() if segment_right is None else segment_right
    )

    features_left_full = (
        df_left.columns.get_level_values("feature").unique().tolist() if features_left is None else features_left
    )
    features_right_full = (
        df_left.columns.get_level_values("feature").unique().tolist() if features_right is None else features_right
    )

    right_value = df_right.loc[timestamps_right_full, pd.IndexSlice[segments_right_full, features_right_full]]
    df_expected.loc[timestamps_left_full, pd.IndexSlice[segments_left_full, features_left_full]] = right_value.values

    df_expected = df_expected.sort_index(axis=1)

    # compare values
    pd.testing.assert_frame_equal(df_obtained, df_expected)


@pytest.mark.parametrize("segments", (["s1"], ["s1", "s2"]))
@pytest.mark.parametrize(
    "columns,answer",
    (
        ({"a", "b"}, set()),
        ({"a", "b", "target"}, {"target"}),
        ({"a", "b", "target", "target_0.5"}, {"target", "target_0.5"}),
        ({"a", "b", "target", "target_0.5", "target1"}, {"target", "target_0.5"}),
        ({"target_component_a", "a", "b", "target_component_c", "target", "target_0.95"}, {"target", "target_0.95"}),
    ),
)
def test_get_target_with_quantiles(segments, columns, answer):
    columns = pd.MultiIndex.from_product([segments, columns], names=["segment", "feature"])
    targets_names = get_target_with_quantiles(columns)
    assert targets_names == answer


@pytest.mark.parametrize(
    "target_level, answer_name",
    (
        ("market", "market_level_constant_forecast_with_quantiles"),
        ("total", "total_level_constant_forecast_with_quantiles"),
    ),
)
def test_get_level_dataframe(product_level_constant_forecast_with_quantiles, target_level, answer_name, request):
    ts = product_level_constant_forecast_with_quantiles
    answer = request.getfixturevalue(answer_name).to_pandas()

    mapping_matrix = ts.hierarchical_structure.get_summing_matrix(
        target_level=target_level, source_level=ts.current_df_level
    )

    target_level_df = get_level_dataframe(
        df=ts.to_pandas(),
        mapping_matrix=mapping_matrix,
        source_level_segments=ts.hierarchical_structure.get_level_segments(level_name=ts.current_df_level),
        target_level_segments=ts.hierarchical_structure.get_level_segments(level_name=target_level),
    )

    pd.testing.assert_frame_equal(target_level_df, answer)


@pytest.mark.parametrize(
    "source_level_segments,target_level_segments,message",
    (
        (("ABC", "c1"), ("X", "Y"), "Segments mismatch for provided dataframe and `source_level_segments`!"),
        (("ABC", "a"), ("X", "Y"), "Segments mismatch for provided dataframe and `source_level_segments`!"),
        (
            ("a", "b", "c", "d"),
            ("X",),
            "Number of target level segments do not match mapping matrix number of columns!",
        ),
    ),
)
def test_get_level_dataframe_segm_errors(
    product_level_simple_hierarchical_ts, source_level_segments, target_level_segments, message
):
    ts = product_level_simple_hierarchical_ts

    mapping_matrix = product_level_simple_hierarchical_ts.hierarchical_structure.get_summing_matrix(
        target_level="market", source_level=ts.current_df_level
    )

    with pytest.raises(ValueError, match=message):
        get_level_dataframe(
            df=ts.df,
            mapping_matrix=mapping_matrix,
            source_level_segments=source_level_segments,
            target_level_segments=target_level_segments,
        )


@pytest.mark.parametrize(
    "features,answer",
    (
        (set(), set()),
        ({"a", "b"}, set()),
        (
            {"target_component_a", "a", "b", "target_component_c", "target", "target_0.95"},
            {"target_component_a", "target_component_c"},
        ),
    ),
)
def test_match_target_components(features, answer):
    components = match_target_components(features)
    assert components == answer


@pytest.fixture
def target_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 1, "target_component_b": 2, "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target_component_a": 3, "target_component_b": 4, "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def inverse_transformed_components_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": [1 * (i + 10) / i for i in range(1, 6)],
            "target_component_b": [2 * (i + 10) / i for i in range(1, 6)],
            "segment": 1,
        }
    )
    df_2 = pd.DataFrame(
        {
            "timestamp": timestamp,
            "target_component_a": [3 * (i + 10) / i for i in range(6, 11)],
            "target_component_b": [4 * (i + 10) / i for i in range(6, 11)],
            "segment": 2,
        }
    )
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def target_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": range(1, 6), "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": range(6, 11), "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


@pytest.fixture
def inverse_transformed_target_df():
    timestamp = pd.date_range("2021-01-01", "2021-01-05")
    df_1 = pd.DataFrame({"timestamp": timestamp, "target": range(11, 16), "segment": 1})
    df_2 = pd.DataFrame({"timestamp": timestamp, "target": range(16, 21), "segment": 2})
    df = pd.concat([df_1, df_2])
    df = TSDataset.to_dataset(df)
    return df


def test_inverse_transform_target_components(
    target_components_df, target_df, inverse_transformed_target_df, inverse_transformed_components_df
):
    obtained_inverse_transformed_components_df = inverse_transform_target_components(
        target_components_df=target_components_df,
        target_df=target_df,
        inverse_transformed_target_df=inverse_transformed_target_df,
    )
    pd.testing.assert_frame_equal(obtained_inverse_transformed_components_df, inverse_transformed_components_df)
