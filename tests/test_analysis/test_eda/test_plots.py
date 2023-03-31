import numpy as np
import pandas as pd
import pytest

from etna.analysis.eda import acf_plot
from etna.analysis.eda.plots import _create_holidays_df
from etna.analysis.eda.plots import _cross_correlation
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df


def test_cross_corr_fail_lengths():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="Lengths of arrays should be equal"):
        _ = _cross_correlation(a=a, b=b)


@pytest.mark.parametrize("max_lags", [-1, 0, 5])
def test_cross_corr_fail_lags(max_lags):
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Parameter maxlags should"):
        _ = _cross_correlation(a=a, b=b, maxlags=max_lags)


@pytest.mark.parametrize("max_lags", [1, 5, 10, 99])
def test_cross_corr_lags(max_lags):
    length = 100
    rng = np.random.default_rng(1)
    a = rng.uniform(low=1.0, high=10.0, size=length)
    b = rng.uniform(low=1.0, high=10.0, size=length)

    result, _ = _cross_correlation(a=a, b=b, maxlags=max_lags)
    expected_result = np.arange(-max_lags, max_lags + 1)

    assert np.all(result == expected_result)


@pytest.mark.parametrize("random_state", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("maxlags", [1, 5, 99])
def test_cross_corr_not_normed(random_state, maxlags):
    length = 100
    rng = np.random.default_rng(random_state)
    a = rng.uniform(low=1.0, high=10.0, size=length)
    b = rng.uniform(low=1.0, high=10.0, size=length)

    _, result = _cross_correlation(a=a, b=b, maxlags=maxlags, normed=False)
    expected_result = np.correlate(a=a, v=b, mode="full")[length - 1 - maxlags : length + maxlags]

    np.testing.assert_almost_equal(result, expected_result)


@pytest.mark.parametrize("random_state", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("maxlags", [1, 5, 99])
def test_cross_corr_not_normed_with_nans(random_state, maxlags):
    length = 100
    rng = np.random.default_rng(random_state)
    a = rng.uniform(low=1.0, high=10.0, size=length)
    b = rng.uniform(low=1.0, high=10.0, size=length)

    fill_nans_a = rng.choice(np.arange(length), replace=False, size=length // 2)
    a[fill_nans_a] = np.NaN

    fill_nans_b = rng.choice(np.arange(length), replace=False, size=length // 2)
    b[fill_nans_b] = np.NaN

    _, result = _cross_correlation(a=a, b=b, maxlags=maxlags, normed=False)
    expected_result = np.correlate(a=np.nan_to_num(a), v=np.nan_to_num(b), mode="full")[
        length - 1 - maxlags : length + maxlags
    ]

    np.testing.assert_almost_equal(result, expected_result)


@pytest.mark.parametrize(
    "a, b, expected_result",
    [
        (np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 8 / np.sqrt(5 * 13), 1.0, 8 / np.sqrt(5 * 13), 1.0]),
        ),
        (np.array([2.0, np.NaN, 2.0]), np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
        (np.array([1.0, np.NaN, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
    ],
)
def test_cross_corr_normed(a, b, expected_result):
    _, result = _cross_correlation(a=a, b=b, normed=True)
    np.testing.assert_almost_equal(result, expected_result)


@pytest.mark.parametrize(
    "a, b, normed, expected_result",
    [
        (np.array([np.NaN, np.NaN, 1.0]), np.array([1.0, 2.0, 3.0]), False, np.array([0.0, 0.0, 3.0, 2.0, 1.0])),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([1.0, 2.0, 3.0]), False, np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
        (np.array([np.NaN, np.NaN, 1.0]), np.array([1.0, 2.0, 3.0]), True, np.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (np.array([np.NaN, np.NaN, np.NaN]), np.array([1.0, 2.0, 3.0]), True, np.array([0.0, 0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_cross_corr_with_full_nans(a, b, normed, expected_result):
    _, result = _cross_correlation(a=a, b=b, maxlags=len(a) - 1, normed=normed)
    np.testing.assert_almost_equal(result, expected_result)


@pytest.fixture
def df_with_nans_in_head(example_df):
    df = TSDataset.to_dataset(example_df)
    df.loc[:4, pd.IndexSlice["segment_1", "target"]] = None
    df.loc[:5, pd.IndexSlice["segment_2", "target"]] = None
    return df


def test_acf_nan_end(ts_diff_endings):
    ts = ts_diff_endings
    acf_plot(ts, partial=False)
    acf_plot(ts, partial=True)


def test_acf_nan_middle(ts_with_nans):
    ts = ts_with_nans
    acf_plot(ts, partial=False)
    with pytest.raises(ValueError):
        acf_plot(ts, partial=True)


def test_acf_nan_begin(df_with_nans_in_head):
    ts = TSDataset(df_with_nans_in_head, freq="H")
    acf_plot(ts, partial=False)
    acf_plot(ts, partial=True)


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
