import pandas as pd
import pytest
from ruptures import Binseg

from etna.transforms.decomposition.change_points_based.change_points_models import RupturesChangePointsModel

N_BKPS = 5


def test_fit_transform_with_nans_in_middle_raise_error(ts_with_nans):
    change_point_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        _ = change_point_model.get_change_points_intervals(df=ts_with_nans.to_pandas()["segment_1"], in_column="target")


def test_get_change_points_intervals_format(simple_ar_df):
    change_point_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    intervals = change_point_model.get_change_points_intervals(df=simple_ar_df, in_column="target")
    assert isinstance(intervals, list)
    assert len(intervals) == N_BKPS + 1
    for interval in intervals:
        assert len(interval) == 2


def test_get_change_points_format(simple_ar_df):
    change_point_model = RupturesChangePointsModel(change_points_model=Binseg(), n_bkps=N_BKPS)
    intervals = change_point_model.get_change_points(df=simple_ar_df, in_column="target")
    assert isinstance(intervals, list)
    assert len(intervals) == N_BKPS
    for interval in intervals:
        assert isinstance(interval, pd.Timestamp)
