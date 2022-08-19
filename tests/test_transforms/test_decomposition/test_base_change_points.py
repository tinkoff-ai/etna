import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.transforms.decomposition.base_change_points import BaseChangePointsModelAdapter
from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel

N_BKPS = 5


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """Generate pd.DataFrame with timestamp."""
    df = pd.DataFrame({"timestamp": pd.date_range("2019-12-01", "2019-12-31")})
    tmp = np.zeros(31)
    tmp[8] = None
    df["target"] = tmp
    df["segment"] = "segment_1"
    df = TSDataset.to_dataset(df=df)
    return df["segment_1"]


@pytest.fixture
def simple_ar_df(random_seed):
    df = generate_ar_df(periods=125, start_time="2021-05-20", n_segments=1, ar_coef=[2], freq="D")
    df_ts_format = TSDataset.to_dataset(df)["segment_0"]
    return df_ts_format


def test_fit_transform_with_nans_in_middle_raise_error(df_with_nans):
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        _ = change_point_model.get_change_points_intervals(df=df_with_nans, in_column="target")


def test_build_intervals():
    """Check correctness of intervals generation with list of change points."""
    change_points = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18"), pd.Timestamp("2020-02-24")]
    expected_intervals = [
        (pd.Timestamp.min, pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-18")),
        (pd.Timestamp("2020-01-18"), pd.Timestamp("2020-02-24")),
        (pd.Timestamp("2020-02-24"), pd.Timestamp.max),
    ]
    intervals = BaseChangePointsModelAdapter._build_intervals(change_points=change_points)
    assert isinstance(intervals, list)
    assert len(intervals) == 4
    for (exp_left, exp_right), (real_left, real_right) in zip(expected_intervals, intervals):
        assert exp_left == real_left
        assert exp_right == real_right


def test_get_change_points_intervals_format(simple_ar_df):
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    intervals = change_point_model.get_change_points_intervals(df=simple_ar_df, in_column="target")
    assert isinstance(intervals, list)
    assert len(intervals) == N_BKPS + 1
    for interval in intervals:
        assert len(interval) == 2


def test_get_change_points_format(simple_ar_df):
    change_point_model = RupturesChangePointsModel(change_point_model=Binseg(), n_bkps=N_BKPS)
    intervals = change_point_model.get_change_points(df=simple_ar_df, in_column="target")
    assert isinstance(intervals, list)
    assert len(intervals) == N_BKPS
    for interval in intervals:
        assert isinstance(interval, pd.Timestamp)
