import pandas as pd
import pytest

from etna.analysis.eda.plots import _create_holidays_df
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


def test_create_holidays_df_str_fail(simple_df):
    with pytest.raises(ValueError):
        _create_holidays_df("RU", simple_df.index, as_is=True)


def test_create_holidays_df_str_non_existing_country(simple_df):
    with pytest.raises((NotImplementedError, KeyError)):
        _create_holidays_df("THIS_COUNTRY_DOES_NOT_EXIST", simple_df.index, as_is=False)


def test_create_holidays_df_str(simple_df):
    df = _create_holidays_df("RU", simple_df.index, as_is=False)
    assert len(df) == len(simple_df.df)
    assert all(df.dtypes == bool)


def test_create_holidays_df_empty_fail(simple_df):
    with pytest.raises(ValueError):
        _create_holidays_df(pd.DataFrame(), simple_df.index, as_is=False)


def test_create_holidays_df_intersect_none(simple_df):
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["1900-01-01", "1901-01-01"])})
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert not df.all(axis=None)


def test_create_holidays_df_one_day(simple_df):
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01"])})
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 1
    assert "New Year" in df.columns


def test_create_holidays_df_upper_window(simple_df):
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01"]), "upper_window": 2})
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 3


def test_create_holidays_df_upper_window_out_of_index(simple_df):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2019-12-25"]), "upper_window": 10})
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 4


def test_create_holidays_df_lower_window(simple_df):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": -2})
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 3


def test_create_holidays_df_lower_window_out_of_index(simple_df):
    holidays = pd.DataFrame(
        {"holiday": "Moscow Anime Festival", "ds": pd.to_datetime(["2020-02-22"]), "lower_window": -5}
    )
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 2


def test_create_holidays_df_lower_upper_windows(simple_df):
    holidays = pd.DataFrame(
        {"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "upper_window": 3, "lower_window": -3}
    )
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 7


def test_create_holidays_df_as_is(simple_df):
    holidays = pd.DataFrame(index=pd.date_range(start="2020-01-07", end="2020-01-10"), columns=["Christmas"], data=1)
    df = _create_holidays_df(holidays, simple_df.index, as_is=True)
    assert df.sum().sum() == 4


def test_create_holidays_df_non_day_freq():
    classic_df = generate_ar_df(periods=30, start_time="2020-01-01", n_segments=1, freq="H")
    ts = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame(
        {
            "holiday": "Christmas",
            "ds": pd.to_datetime(
                ["2020-01-01"],
            ),
            "upper_window": 3,
        }
    )
    df = _create_holidays_df(holidays, ts.index, as_is=False)
    assert df.sum().sum() == 4


def test_create_holidays_df_15t_freq():
    classic_df = generate_ar_df(periods=30, start_time="2020-01-01", n_segments=1, freq="15T")
    ts = TSDataset.to_dataset(classic_df)
    holidays = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01 01:00:00"]), "upper_window": 3})
    df = _create_holidays_df(holidays, ts.index, as_is=False)
    assert df.sum().sum() == 4
    assert df.loc["2020-01-01 01:00:00":"2020-01-01 01:45:00"].sum().sum() == 4


def test_create_holidays_df_several_holidays(simple_df):
    christmas = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": -3})
    new_year = pd.DataFrame({"holiday": "New Year", "ds": pd.to_datetime(["2020-01-01"]), "upper_window": 2})
    holidays = pd.concat((christmas, new_year))
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 7


def test_create_holidays_df_zero_windows(simple_df):
    holidays = pd.DataFrame(
        {"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": 0, "upper_window": 0}
    )
    df = _create_holidays_df(holidays, simple_df.index, as_is=False)
    assert df.sum().sum() == 1
    assert df.loc["2020-01-07"].sum() == 1


def test_create_holidays_df_upper_window_negative(simple_df):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "upper_window": -1})
    with pytest.raises(ValueError):
        _create_holidays_df(holidays, simple_df.index, as_is=False)


def test_create_holidays_df_lower_window_positive(simple_df):
    holidays = pd.DataFrame({"holiday": "Christmas", "ds": pd.to_datetime(["2020-01-07"]), "lower_window": 1})
    with pytest.raises(ValueError):
        _create_holidays_df(holidays, simple_df.index, as_is=False)
