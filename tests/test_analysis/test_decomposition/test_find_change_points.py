from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.analysis import find_change_points
from etna.datasets import TSDataset


def check_change_points(change_points: Dict[str, List[pd.Timestamp]], segments: List[str], num_points: int):
    """Check change points on validity."""
    assert isinstance(change_points, dict)
    assert set(change_points.keys()) == set(segments)
    for segment in segments:
        change_points_segment = change_points[segment]
        assert len(change_points_segment) == num_points
        for point in change_points_segment:
            assert isinstance(point, pd.Timestamp)


@pytest.mark.parametrize("n_bkps", [5, 10, 12, 27])
def test_find_change_points_simple(multitrend_df: pd.DataFrame, n_bkps: int):
    """Test that find_change_points works fine with multitrend example."""
    ts = TSDataset(df=multitrend_df, freq="D")
    change_points = find_change_points(ts=ts, in_column="target", change_point_model=Binseg(), n_bkps=n_bkps)
    check_change_points(change_points, segments=ts.segments, num_points=n_bkps)


@pytest.mark.parametrize("n_bkps", [5, 10, 12, 27])
def test_find_change_points_nans_head(multitrend_df: pd.DataFrame, n_bkps: int):
    """Test that find_change_points works fine with nans at the beginning of the series."""
    multitrend_df.iloc[:5, :] = np.NaN
    ts = TSDataset(df=multitrend_df, freq="D")
    change_points = find_change_points(ts=ts, in_column="target", change_point_model=Binseg(), n_bkps=n_bkps)
    check_change_points(change_points, segments=ts.segments, num_points=n_bkps)


@pytest.mark.parametrize("n_bkps", [5, 10, 12, 27])
def test_find_change_points_nans_tail(multitrend_df: pd.DataFrame, n_bkps: int):
    """Test that find_change_points works fine with nans at the end of the series."""
    multitrend_df.iloc[-5:, :] = np.NaN
    ts = TSDataset(df=multitrend_df, freq="D")
    change_points = find_change_points(ts=ts, in_column="target", change_point_model=Binseg(), n_bkps=n_bkps)
    check_change_points(change_points, segments=ts.segments, num_points=n_bkps)
