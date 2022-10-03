import numpy as np
import pytest

from etna.analysis.outliers import get_anomalies_median


def test_const_ts(const_ts_anomal):
    anomal = get_anomalies_median(const_ts_anomal)
    assert {"segment_0", "segment_1"} == set(anomal.keys())
    for seg in anomal.keys():
        assert len(anomal[seg]) == 0


@pytest.mark.parametrize(
    "window_size, alpha, right_anomal",
    (
        (10, 3, {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]}),
        (
            10,
            2,
            {
                "1": [np.datetime64("2021-01-11")],
                "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-16"), np.datetime64("2021-01-27")],
            },
        ),
        (20, 2, {"1": [np.datetime64("2021-01-11")], "2": [np.datetime64("2021-01-09"), np.datetime64("2021-01-27")]}),
    ),
)
def test_median_outliers(window_size, alpha, right_anomal, outliers_tsds):
    assert get_anomalies_median(ts=outliers_tsds, window_size=window_size, alpha=alpha) == right_anomal


@pytest.mark.parametrize("true_params", (["1", "2"],))
def test_interface_correct_args(true_params, outliers_tsds):
    d = get_anomalies_median(ts=outliers_tsds, window_size=10, alpha=2)
    assert isinstance(d, dict)
    assert sorted(d.keys()) == sorted(true_params)
    for i in d.keys():
        for j in d[i]:
            assert isinstance(j, np.datetime64)


def test_in_column(outliers_df_with_two_columns):
    outliers = get_anomalies_median(ts=outliers_df_with_two_columns, in_column="feature", window_size=10)
    expected = {"1": [np.datetime64("2021-01-08")], "2": [np.datetime64("2021-01-26")]}
    for key in expected:
        assert key in outliers
        np.testing.assert_array_equal(outliers[key], expected[key])
