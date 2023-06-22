from datetime import datetime

import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.timestamp import SpecialDaysTransform
from etna.transforms.timestamp.special_days import _OneSegmentSpecialDaysTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


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
def ts_with_specials(df_with_specials):
    """Create dataset with special weekdays and monthdays."""
    df = df_with_specials.reset_index()
    df["segment"] = "1"
    df = df[["timestamp", "segment", "target"]]
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D")
    return ts


@pytest.fixture()
def constant_days_two_segments_ts(constant_days_df: pd.DataFrame):
    """Create TSDataset that has two segments with constant columns each."""
    df_1 = constant_days_df.reset_index()
    df_2 = constant_days_df.reset_index()

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = TSDataset.to_dataset(classic_df)
    ts = TSDataset(df=df, freq="D")
    return ts


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


def test_interface_two_segments_week(constant_days_two_segments_ts: TSDataset):
    """
    This test checks that SpecialDaysTransform that should find special weekdays creates the only column with
    'anomaly_weekdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=False)
    df = special_days_finder.fit_transform(constant_days_two_segments_ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" in df[segment].columns
        assert "anomaly_monthdays" not in df[segment].columns
        assert df[segment]["anomaly_weekdays"].dtype == "category"


def test_interface_two_segments_month(constant_days_two_segments_ts: TSDataset):
    """
    This test checks that SpecialDaysTransform that should find special month days creates the only column with
    'anomaly_monthdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=False, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_two_segments_ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" not in df[segment].columns
        assert "anomaly_monthdays" in df[segment].columns
        assert df[segment]["anomaly_monthdays"].dtype == "category"


def test_interface_two_segments_week_month(constant_days_two_segments_ts: TSDataset):
    """
    This test checks that SpecialDaysTransform that should find special month and week days
    creates two columns with 'anomaly_monthdays' and 'anomaly_weekdays' name as expected.
    """
    special_days_finder = SpecialDaysTransform(find_special_weekday=True, find_special_month_day=True)
    df = special_days_finder.fit_transform(constant_days_two_segments_ts).to_pandas()
    for segment in df.columns.get_level_values("segment").unique():
        assert "anomaly_weekdays" in df[segment].columns
        assert "anomaly_monthdays" in df[segment].columns
        assert df[segment]["anomaly_weekdays"].dtype == "category"
        assert df[segment]["anomaly_monthdays"].dtype == "category"


def test_interface_two_segments_noweek_nomonth():
    """This test checks that bad-inited SpecialDaysTransform raises ValueError on initialisation."""
    with pytest.raises(
        ValueError, match="_OneSegmentSpecialDaysTransform feature does nothing with given init args configuration"
    ):
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
    transform.fit_transform(ts_diff_endings)


def test_save_load(ts_with_specials):
    ts = ts_with_specials
    transform = SpecialDaysTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=ts)


def test_params_to_tune(ts_with_specials):
    def skip_parameters(parameters):
        names = ["find_special_weekday", "find_special_month_day"]
        values = [not parameters[x] for x in names]
        if all(values):
            return True
        return False

    transform = SpecialDaysTransform()
    ts = ts_with_specials
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts, skip_parameters=skip_parameters)
