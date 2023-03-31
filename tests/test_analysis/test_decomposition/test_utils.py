import numpy as np
import pandas as pd
import pytest

from etna.analysis.decomposition.utils import _get_labels_names
from etna.analysis.decomposition.utils import _resample
from etna.analysis.decomposition.utils import _seasonal_split
from etna.datasets import TSDataset
from etna.transforms import LinearTrendTransform
from etna.transforms import TheilSenTrendTransform


@pytest.mark.parametrize(
    "poly_degree, expect_values, trend_class",
    (
        [1, True, LinearTrendTransform],
        [2, False, LinearTrendTransform],
        [1, True, TheilSenTrendTransform],
        [2, False, TheilSenTrendTransform],
    ),
)
def test_get_labels_names_linear_coeffs(example_tsdf, poly_degree, expect_values, trend_class):
    ln_tr = trend_class(in_column="target", poly_degree=poly_degree)
    ln_tr.fit_transform(example_tsdf)
    segments = example_tsdf.segments
    _, linear_coeffs = _get_labels_names([ln_tr], segments)
    if expect_values:
        assert list(linear_coeffs.values()) != ["", ""]
    else:
        assert list(linear_coeffs.values()) == ["", ""]


@pytest.mark.parametrize(
    "timestamp, cycle, expected_cycle_names, expected_in_cycle_nums, expected_in_cycle_names",
    [
        (
            pd.date_range(start="2020-01-01", periods=5, freq="D"),
            3,
            ["1", "1", "1", "2", "2"],
            [0, 1, 2, 0, 1],
            ["0", "1", "2", "0", "1"],
        ),
        (
            pd.date_range(start="2020-01-01", periods=6, freq="15T"),
            "hour",
            ["2020-01-01 00"] * 4 + ["2020-01-01 01"] * 2,
            [0, 1, 2, 3, 0, 1],
            ["0", "1", "2", "3", "0", "1"],
        ),
        (
            pd.date_range(start="2020-01-01", periods=26, freq="H"),
            "day",
            ["2020-01-01"] * 24 + ["2020-01-02"] * 2,
            [i % 24 for i in range(26)],
            [str(i % 24) for i in range(26)],
        ),
        (
            pd.date_range(start="2020-01-01", periods=10, freq="D"),
            "week",
            ["2020-00"] * 5 + ["2020-01"] * 5,
            [2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
            ["Wed", "Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        ),
        (
            pd.date_range(start="2020-01-03", periods=40, freq="D"),
            "month",
            ["2020-Jan"] * 29 + ["2020-Feb"] * 11,
            list(range(3, 32)) + list(range(1, 12)),
            [str(i) for i in range(3, 32)] + [str(i) for i in range(1, 12)],
        ),
        (
            pd.date_range(start="2020-01-01", periods=14, freq="M"),
            "quarter",
            ["2020-1"] * 3 + ["2020-2"] * 3 + ["2020-3"] * 3 + ["2020-4"] * 3 + ["2021-1"] * 2,
            [i % 3 for i in range(14)],
            [str(i % 3) for i in range(14)],
        ),
        (
            pd.date_range(start="2020-01-01", periods=14, freq="M"),
            "year",
            ["2020"] * 12 + ["2021"] * 2,
            [i % 12 + 1 for i in range(14)],
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"],
        ),
    ],
)
def test_seasonal_split(timestamp, cycle, expected_cycle_names, expected_in_cycle_nums, expected_in_cycle_names):
    cycle_df = _seasonal_split(timestamp=timestamp.to_series(), freq=timestamp.freq.freqstr, cycle=cycle)
    assert cycle_df["cycle_name"].tolist() == expected_cycle_names
    assert cycle_df["in_cycle_num"].tolist() == expected_in_cycle_nums
    assert cycle_df["in_cycle_name"].tolist() == expected_in_cycle_names


@pytest.mark.parametrize(
    "timestamp, values, resample_freq, aggregation, expected_timestamp, expected_values",
    [
        (
            pd.date_range(start="2020-01-01", periods=14, freq="Q"),
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 10, 16, 10, 5, 7, 5, 7, 3, 3],
            "Y",
            "sum",
            pd.date_range(start="2020-01-01", periods=4, freq="Y"),
            [np.NaN, 36.0, 24.0, 6.0],
        ),
        (
            pd.date_range(start="2020-01-01", periods=14, freq="Q"),
            [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 10, 16, 10, 5, 7, 5, 7, 3, 3],
            "Y",
            "mean",
            pd.date_range(start="2020-01-01", periods=4, freq="Y"),
            [np.NaN, 12.0, 6.0, 3.0],
        ),
    ],
)
def test_resample(timestamp, values, resample_freq, aggregation, expected_timestamp, expected_values):
    df = pd.DataFrame({"timestamp": timestamp.tolist(), "target": values, "segment": len(timestamp) * ["segment_0"]})
    df_wide = TSDataset.to_dataset(df)
    df_resampled = _resample(df=df_wide, freq=resample_freq, aggregation=aggregation)
    assert df_resampled.index.tolist() == expected_timestamp.tolist()
    assert (
        df_resampled.loc[:, pd.IndexSlice["segment_0", "target"]]
        .reset_index(drop=True)
        .equals(pd.Series(expected_values))
    )
