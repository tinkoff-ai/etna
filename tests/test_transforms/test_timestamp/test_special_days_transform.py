from datetime import datetime

import pandas as pd
import pytest

from etna.transforms.timestamp import SpecialDaysTransform
from etna.transforms.timestamp.special_days import _OneSegmentSpecialDaysTransform


@pytest.fixture()
def constant_days_df():
    """Create pandas dataframe that represents one segment and has const value column"""
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-01", end="2020-04-01", freq="D")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def df_with_specials():
    """Create pandas dataframe that represents one segment and has non-const value column."""
    weekday_outliers_dates = [
        # monday
        {"timestamp": datetime(2020, 12, 28).date(), "target": 10},
        # tuesday
        {"timestamp": datetime(2020, 1, 7).date(), "target": 20},
        # wednesdays
        {"timestamp": datetime(2020, 2, 12).date(), "target": 5},
        {"timestamp": datetime(2020, 9, 30).date(), "target": 10},
        {"timestamp": datetime(2020, 6, 10).date(), "target": 14},
        # sunday
        {"timestamp": datetime(2020, 5, 10).date(), "target": 12},
    ]
    special_df = pd.DataFrame(weekday_outliers_dates)
    special_df["timestamp"] = pd.to_datetime(special_df["timestamp"])
    date_range = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", "2020-12-31")})
    df = pd.merge(date_range, special_df, on="timestamp", how="left").fillna(0)

    special_weekdays = (2,)
    special_monthdays = (7, 10)

    df["week_true"] = df["timestamp"].apply(lambda x: x.weekday() in special_weekdays)
    df["month_true"] = df["timestamp"].apply(lambda x: x.day in special_monthdays)
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def constant_days_two_segments_df(constant_days_df: pd.DataFrame):
    """Create pandas dataframe that has two segments with constant columns each."""
    df_1 = constant_days_df.reset_index()
    df_2 = constant_days_df.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


def test_interface_week(constant_days_df: pd.DataFrame):
    """
    This test checks that _OneSegmentSpecialDaysTransform that should find special weekdays creates the only column with
    'anomaly_weekdays' name as expected.
    """
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=True, find_special_month_day=False)
    df = special_days_finder.fit_transform(constant_days_df)
    assert "anomaly_weekdays" in df.columns
    assert "anomaly_monthdays" not in df.columns
    assert df["anomaly_weekdays"].dtype == "category"


def test_interface_month(constant_days_df: pd.DataFrame):
    """
    This test checks that _OneSegmentSpecialDaysTransform that should find special month days creates the only column with
    'anomaly_monthdays' name as expected.
    """
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=False, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_df)
    assert "anomaly_weekdays" not in df.columns
    assert "anomaly_monthdays" in df.columns
    assert df["anomaly_monthdays"].dtype == "category"


def test_interface_week_month(constant_days_df: pd.DataFrame):
    """
    This test checks that _OneSegmentSpecialDaysTransform that should find special month and week days
    creates two columns with 'anomaly_monthdays' and 'anomaly_weekdays' name as expected.
    """
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_df)
    assert "anomaly_weekdays" in df.columns
    assert "anomaly_monthdays" in df.columns
    assert df["anomaly_weekdays"].dtype == "category"
    assert df["anomaly_monthdays"].dtype == "category"


def test_interface_noweek_nomonth():
    """This test checks that bad-inited _OneSegmentSpecialDaysTransform raises AssertionError."""
    with pytest.raises(ValueError):
        _ = _OneSegmentSpecialDaysTransform(find_special_weekday=False, find_special_month_day=False)


def test_interface_two_segments_week(constant_days_two_segments_df: pd.DataFrame):
    """
    This test checks that SpecialDaysTransform that should find special weekdays creates the only column with
    'anomaly_weekdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=False)
    df = special_days_finder.fit_transform(constant_days_two_segments_df)
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" in df[segment].columns
        assert "anomaly_monthdays" not in df[segment].columns
        assert df[segment]["anomaly_weekdays"].dtype == "category"


def test_interface_two_segments_month(constant_days_two_segments_df: pd.DataFrame):
    """
    This test checks that SpecialDaysTransform that should find special month days creates the only column with
    'anomaly_monthdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=False, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_two_segments_df)
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" not in df[segment].columns
        assert "anomaly_monthdays" in df[segment].columns
        assert df[segment]["anomaly_monthdays"].dtype == "category"


def test_interface_two_segments_week_month(constant_days_two_segments_df: pd.DataFrame):
    """
    This test checks that SpecialDaysTransform that should find special month and week days
    creates two columns with 'anomaly_monthdays' and 'anomaly_weekdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_two_segments_df)
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" in df[segment].columns
        assert "anomaly_monthdays" in df[segment].columns
        assert df[segment]["anomaly_weekdays"].dtype == "category"
        assert df[segment]["anomaly_monthdays"].dtype == "category"


def test_interface_two_segments_noweek_nomonth(constant_days_two_segments_df: pd.DataFrame):
    """This test checks that bad-inited SpecialDaysTransform raises AssertionError during fit_transform."""
    with pytest.raises(ValueError):
        _ = SpecialDaysTransform(find_special_weekday=False, find_special_month_day=False)


def test_week_feature(df_with_specials: pd.DataFrame):
    """This test checks that _OneSegmentSpecialDaysTransform computes weekday feature correctly."""
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=True, find_special_month_day=False)
    df = special_days_finder.fit_transform(df_with_specials)
    assert (df_with_specials["week_true"] == df["anomaly_weekdays"]).all()


def test_month_feature(df_with_specials: pd.DataFrame):
    """This test checks that _OneSegmentSpecialDaysTransform computes monthday feature correctly."""
    special_days_finder = _OneSegmentSpecialDaysTransform(find_special_weekday=False, find_special_month_day=True)
    df = special_days_finder.fit_transform(df_with_specials)
    assert (df_with_specials["month_true"] == df["anomaly_monthdays"]).all()


def test_no_false_positive_week(constant_days_df: pd.DataFrame):
    """This test checks that there is no false-positive results in week mode."""
    special_days_finder = _OneSegmentSpecialDaysTransform()
    res = special_days_finder.fit_transform(constant_days_df)
    assert res["anomaly_weekdays"].astype("bool").sum() == 0


def test_no_false_positive_month(constant_days_df: pd.DataFrame):
    """This test checks that there is no false-positive results in month mode."""
    special_days_finder = _OneSegmentSpecialDaysTransform()
    res = special_days_finder.fit_transform(constant_days_df)
    assert res["anomaly_monthdays"].astype("bool").sum() == 0


def test_transform_raise_error_if_not_fitted(constant_days_df: pd.DataFrame):
    """Test that transform for one segment raise error when calling transform without being fit."""
    transform = _OneSegmentSpecialDaysTransform()
    with pytest.raises(ValueError, match="Transform is not fitted!"):
        _ = transform.transform(df=constant_days_df)


def test_fit_transform_with_nans(ts_diff_endings):
    transform = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    ts_diff_endings.fit_transform([transform])
