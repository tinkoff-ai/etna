import numpy as np
import pytest

from etna.analysis.outliers import get_anomalies_median


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
    assert get_anomalies_median(outliers_tsds, window_size, alpha) == right_anomal


@pytest.mark.parametrize("true_params", (["1", "2"],))
def test_interface_correct_args(true_params, outliers_tsds):
    d = get_anomalies_median(outliers_tsds, 10, 2)
    assert isinstance(d, dict)
    assert sorted(list(d.keys())) == sorted(true_params)
    for i in d.keys():
        for j in d[i]:
            assert isinstance(j, np.datetime64)
