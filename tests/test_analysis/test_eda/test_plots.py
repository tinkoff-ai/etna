import numpy as np
import pandas as pd
import pytest

from etna.analysis.eda import acf_plot
from etna.analysis.eda.plots import _cross_correlation
from etna.datasets import TSDataset


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
