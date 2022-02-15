import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.transforms.timestamp import HolidayTransform


@pytest.fixture()
def simple_ts_with_regressors():
    df = generate_const_df(scale=1, n_segments=3, start_time="2020-01-01", periods=100)
    df_exog = generate_const_df(scale=10, n_segments=3, start_time="2020-01-01", periods=150).rename(
        {"target": "regressor_a"}, axis=1
    )
    ts = TSDataset(df=TSDataset.to_dataset(df), freq="D", df_exog=TSDataset.to_dataset(df_exog))
    return ts


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
    df = TSDataset.to_dataset(classic_df)
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
    df = TSDataset.to_dataset(classic_df)
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
    df = TSDataset.to_dataset(classic_df)
    return df


def test_holiday_with_regressors(simple_ts_with_regressors: TSDataset):
    simple_ts_with_regressors.fit_transform([HolidayTransform(out_column="holiday")])
    len_holiday = len([cols for cols in simple_ts_with_regressors.columns if cols[1] == "holiday"])
    assert len_holiday == len(np.unique(simple_ts_with_regressors.columns.get_level_values("segment")))


def test_interface_two_segments_daily(two_segments_simple_df_daily: pd.DataFrame):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    df = holidays_finder.fit_transform(two_segments_simple_df_daily)
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


def test_interface_two_segments_hour(two_segments_simple_df_hour: pd.DataFrame):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    df = holidays_finder.fit_transform(two_segments_simple_df_hour)
    for segment in df.columns.get_level_values("segment").unique():
        assert "regressor_holidays" in df[segment].columns
        assert df[segment]["regressor_holidays"].dtype == "category"


def test_interface_two_segments_min(two_segments_simple_df_min: pd.DataFrame):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
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
def test_holidays_day(iso_code: str, answer: np.array, two_segments_simple_df_daily: pd.DataFrame):
    holidays_finder = HolidayTransform(iso_code=iso_code, out_column="regressor_holidays")
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
def test_holidays_hour(iso_code: str, answer: np.array, two_segments_simple_df_hour: pd.DataFrame):
    holidays_finder = HolidayTransform(iso_code=iso_code, out_column="regressor_holidays")
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
def test_holidays_min(iso_code: str, answer: np.array, two_segments_simple_df_min: pd.DataFrame):
    holidays_finder = HolidayTransform(iso_code=iso_code, out_column="regressor_holidays")
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
def test_holidays_failed(index: pd.DatetimeIndex, two_segments_simple_df_daily: pd.DataFrame):
    df = two_segments_simple_df_daily
    df.index = index
    holidays_finder = HolidayTransform()
    with pytest.raises(ValueError, match="Frequency of data should be no more than daily."):
        df = holidays_finder.fit_transform(df)


@pytest.mark.parametrize("expected_regressors", ([["regressor_holidays"]]))
def test_holidays_out_column_added_to_regressors(example_tsds, expected_regressors):
    holidays_finder = HolidayTransform(out_column="regressor_holidays")
    example_tsds.fit_transform([holidays_finder])
    assert sorted(example_tsds.regressors) == sorted(expected_regressors)
