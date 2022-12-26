import pandas as pd

from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter


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
