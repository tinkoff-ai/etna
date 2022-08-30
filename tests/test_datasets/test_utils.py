from itertools import chain
from itertools import permutations
from itertools import product

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data
from etna.datasets import generate_ar_df
from etna.datasets.utils import _TorchDataset
from etna.datasets.utils import get_loc_wide


@pytest.fixture
def df_exog_no_segments() -> pd.DataFrame:
    timestamp = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"timestamp": timestamp, "exog_1": 1, "exog_2": 2, "exog_3": 3})
    return df


@pytest.fixture
def df_wide() -> pd.DataFrame:
    df = generate_ar_df(periods=5, start_time="2020-01-01", n_segments=3)
    df_wide = TSDataset.to_dataset(df)

    df_exog = df.copy()
    df_exog = df_exog.rename(columns={"target": "exog_1"})
    df_exog["exog_1"] = df_exog["exog_1"] + 1
    df_exog["exog_2"] = df_exog["exog_1"] + 1
    df_exog["exog_3"] = df_exog["exog_2"] + 1
    df_exog_wide = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq="D")
    df = ts.df

    # make some reorderings for checking corner cases
    df = df.loc[:, pd.IndexSlice[["segment_2", "segment_0", "segment_1"], ["target", "exog_3", "exog_2", "exog_1"]]]

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


def _check_get_loc_wide(df_wide, timestamps, segments, features):
    obtained_df = get_loc_wide(df_wide, timestamps, segments, features)

    if timestamps is None:
        timestamps = df_wide.index

    if segments is None:
        segments = df_wide.columns.get_level_values("segment").unique()

    if features is None:
        features = df_wide.columns.get_level_values("feature").unique()

    expected_df = df_wide.loc[timestamps, pd.IndexSlice[segments, features]]

    try:
        pd.testing.assert_frame_equal(obtained_df, expected_df)
    except AssertionError:
        x = 10


def all_subsets(iterable):
    s = list(iterable)
    return chain.from_iterable(permutations(s, r) for r in range(1, len(s) + 1))


def test_get_loc_wide(df_wide):
    possible_timestamps = chain(all_subsets(df_wide.index.tolist()), [None])
    possible_segments = chain(all_subsets(df_wide.columns.get_level_values("segment").unique().tolist()), [None])
    possible_features = chain(all_subsets(df_wide.columns.get_level_values("feature").unique().tolist()), [None])
    iterator = product(possible_timestamps, possible_segments, possible_features)

    idx = 0

    for timestamps, segments, features in iterator:
        _check_get_loc_wide(df_wide, timestamps, segments, features)
        idx += 1

    x = 10
