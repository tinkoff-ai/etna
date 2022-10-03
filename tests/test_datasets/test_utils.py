import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data
from etna.datasets import generate_ar_df
from etna.datasets.utils import _TorchDataset
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
