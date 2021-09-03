from typing import List
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from etna.datasets.tsdataset import TSDataset
from etna.transforms.lags import LagTransform
from etna.transforms.lags import _OneSegmentLagFeature


@pytest.fixture
def int_df_one_segment() -> pd.DataFrame:
    """Generate dataframe with simple targets for lags check."""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-06-01")})
    df["segment"] = "segment_1"
    df["target"] = np.arange(0, len(df))
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def int_df_two_segments(int_df_one_segment) -> pd.DataFrame:
    """Generate dataframe with simple targets for lags check."""
    df_1 = int_df_one_segment.reset_index()
    df_2 = int_df_one_segment.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    df = pd.concat([df_1, df_2], ignore_index=True)
    return TSDataset.to_dataset(df)


@pytest.mark.parametrize(
    "lags,expected_columns",
    ((4, ["target_lag_1", "target_lag_2", "target_lag_3", "target_lag_4"]), ([5, 8], ["target_lag_5", "target_lag_8"])),
)
def test_interface_one_segment(
    lags: Union[int, Sequence[int]], expected_columns: List[str], int_df_one_segment: pd.DataFrame
):
    """This test checks _OneSegmentLagFeature interface."""
    lf = _OneSegmentLagFeature(in_column="target", lags=lags)
    lags_df = lf.fit_transform(df=int_df_one_segment)
    lags_df_lags_columns = sorted(filter(lambda x: x.startswith("target_lag"), lags_df.columns))
    assert lags_df_lags_columns == expected_columns


@pytest.mark.parametrize(
    "lags,expected_columns",
    ((4, ["target_lag_1", "target_lag_2", "target_lag_3", "target_lag_4"]), ([5, 8], ["target_lag_5", "target_lag_8"])),
)
def test_interface_two_segments(lags: Union[int, Sequence[int]], expected_columns: List[str], int_df_two_segments):
    """This test checks LagTransform interface."""
    lf = LagTransform(in_column="target", lags=lags)
    lags_df = lf.fit_transform(df=int_df_two_segments)
    for segment in lags_df.columns.get_level_values("segment").unique():
        lags_df_lags_columns = sorted(filter(lambda x: x.startswith("target_lag"), lags_df[segment].columns))
        assert lags_df_lags_columns == expected_columns


@pytest.mark.parametrize("lags", (12, [4, 6, 8, 16]))
def test_lags_values_one_segment(lags: Union[int, Sequence[int]], int_df_one_segment: pd.DataFrame):
    """This test checks that _OneSegmentLagFeature computes lags correctly."""
    lf = _OneSegmentLagFeature(in_column="target", lags=lags)
    lags_df = lf.fit_transform(df=int_df_one_segment)
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    for lag in lags:
        true_values = pd.Series([None] * lag + list(int_df_one_segment["target"].values[:-lag]))
        assert_almost_equal(true_values.values, lags_df[f"target_lag_{lag}"].values)


@pytest.mark.parametrize("lags", (12, [4, 6, 8, 16]))
def test_lags_values_two_segments(lags: Union[int, Sequence[int]], int_df_two_segments):
    """This test checks that LagTransform computes lags correctly."""
    lf = LagTransform(in_column="target", lags=lags)
    lags_df = lf.fit_transform(df=int_df_two_segments)
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    for segment in lags_df.columns.get_level_values("segment").unique():
        for lag in lags:
            true_values = pd.Series([None] * lag + list(int_df_two_segments[segment, "target"].values[:-lag]))
            assert_almost_equal(true_values.values, lags_df[segment, f"target_lag_{lag}"].values)


@pytest.mark.parametrize("lags", (0, -1, (10, 15, -2)))
def test_invalid_lags_value_one_segment(lags):
    """This test check _OneSegmentLagFeature's behavior in case of invalid set of params."""
    with pytest.raises(ValueError):
        _ = _OneSegmentLagFeature(in_column="target", lags=lags)


@pytest.mark.parametrize("lags", (0, -1, (10, 15, -2)))
def test_invalid_lags_value_two_segments(lags):
    """This test check LagTransform's behavior in case of invalid set of params."""
    with pytest.raises(ValueError):
        _ = LagTransform(in_column="target", lags=lags)
