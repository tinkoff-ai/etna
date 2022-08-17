import numpy as np
import pandas as pd
import pytest
from ruptures import Binseg

from etna.datasets import TSDataset
from etna.transforms.decomposition.base_change_points import BaseChangePointsModelAdapter
from etna.transforms.decomposition.base_change_points import RupturesChangePointsModel


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


def test_fit_transform_with_nans_in_middle_raise_error(df_with_nans):
    change_point_model = RupturesChangePointsModel(Binseg(), n_bkps=5)
    with pytest.raises(ValueError, match="The input column contains NaNs in the middle of the series!"):
        change_point_model.get_change_points_intervals(df=df_with_nans, in_column="target")
