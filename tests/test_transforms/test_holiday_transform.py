import numpy as np
import pandas as pd
import pytest

from etna.transforms.holiday import HolidayTransform


@pytest.fixture()
def simple_constant_df_daily():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_df_daily(simple_constant_df_daily: pd.DataFrame):
    df_1 = simple_constant_df_daily.reset_index()
    df_2 = simple_constant_df_daily.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


@pytest.fixture()
def simple_constant_df_hour():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-01-08 22:15", end="2020-01-10", freq="H")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_df_hour(simple_constant_df_hour: pd.DataFrame):
    df_1 = simple_constant_df_hour.reset_index()
    df_2 = simple_constant_df_hour.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


@pytest.fixture()
def simple_constant_df_min():
    df = pd.DataFrame({"timestamp": pd.date_range(start="2020-11-25 22:30", end="2020-11-26 02:15", freq="15MIN")})
    df["target"] = 42
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture()
def two_segments_simple_df_min(simple_constant_df_min: pd.DataFrame):
    df_1 = simple_constant_df_min.reset_index()
    df_2 = simple_constant_df_min.reset_index()
    df_1 = df_1[3:]

    df_1["segment"] = "segment_1"
    df_2["segment"] = "segment_2"

    classic_df = pd.concat([df_1, df_2], ignore_index=True)
    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


def test_interface_two_segments_daily(two_segments_simple_df_daily: pd.DataFrame):
    holidays_finder = HolidayTransform()
    df = holidays_finder.fit_transform(two_segments_simple_df_daily)
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


def test_interface_two_segments_hour(two_segments_simple_df_hour: pd.DataFrame):
    holidays_finder = HolidayTransform()
    df = holidays_finder.fit_transform(two_segments_simple_df_hour)
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


def test_interface_two_segments_min(two_segments_simple_df_min: pd.DataFrame):
    holidays_finder = HolidayTransform()
    df = holidays_finder.fit_transform(two_segments_simple_df_min)
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_holidays_day(iso_code: str, answer: np.array, two_segments_simple_df_daily):
    holidays_finder = HolidayTransform(iso_code)
    df = holidays_finder.fit_transform(two_segments_simple_df_daily)
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, answer)


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ),
)
def test_holidays_hour(iso_code: str, answer: np.array, two_segments_simple_df_hour):
    holidays_finder = HolidayTransform(iso_code)
    df = holidays_finder.fit_transform(two_segments_simple_df_hour)
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, answer)


@pytest.mark.parametrize(
    "iso_code,answer",
    (
        ("RUS", np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("US", np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ),
)
def test_holidays_min(iso_code: str, answer: np.array, two_segments_simple_df_min):
    holidays_finder = HolidayTransform(iso_code)
    df = holidays_finder.fit_transform(two_segments_simple_df_min)
    for segment in df.columns.get_level_values("segment").unique():
        assert np.array_equal(df[segment]["regressor_holidays"].values, answer)


@pytest.mark.parametrize(
    "index",
    (
        (pd.date_range(start="2020-11-25 22:30", end="2020-12-11", freq="1D 15MIN")),
        (pd.date_range(start="2019-11-25", end="2021-02-25", freq="M")),
    ),
)
def holidays_failed(index, two_segments_simple_df_daily):
    df = two_segments_simple_df_daily
    df["index"] = index
    holidays_finder = HolidayTransform()
    with pytest.raises(ValueError, match="Frequency of data should be no more than daily."):
        df = holidays_finder.fit_transform(df)
