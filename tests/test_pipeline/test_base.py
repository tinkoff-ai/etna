import pandas as pd
import pytest

from etna.pipeline.base import BasePipeline


@pytest.mark.parametrize(
    "ts_name, expected_start_timestamp, expected_end_timestamp",
    [
        ("example_tsds", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-09")),
        ("ts_with_different_series_length", pd.Timestamp("2020-01-01 4:00"), pd.Timestamp("2020-02-01")),
    ],
)
def test_make_predict_timestamps_calculate_values(ts_name, expected_start_timestamp, expected_end_timestamp, request):
    ts = request.getfixturevalue(ts_name)

    start_timestamp, end_timestamp = BasePipeline._make_predict_timestamps(ts=ts)

    assert start_timestamp == expected_start_timestamp
    assert end_timestamp == expected_end_timestamp


def test_make_predict_timestamps_fail_early_start(example_tsds):
    start_timestamp = example_tsds.index[0] - pd.DateOffset(days=5)
    with pytest.raises(ValueError, match="Value of start_timestamp is less than beginning of some segments"):
        _ = BasePipeline._make_predict_timestamps(ts=example_tsds, start_timestamp=start_timestamp)


def test_make_predict_timestamps_fail_late_end(example_tsds):
    end_timestamp = example_tsds.index[-1] + pd.DateOffset(days=5)
    with pytest.raises(ValueError, match="Value of end_timestamp is more than ending of dataset"):
        _ = BasePipeline._make_predict_timestamps(ts=example_tsds, end_timestamp=end_timestamp)


def test_make_predict_timestamps_fail_start_later_than_end(example_tsds):
    start_timestamp = example_tsds.index[2]
    end_timestamp = example_tsds.index[0]
    with pytest.raises(ValueError, match="Value of end_timestamp is less than start_timestamp"):
        _ = BasePipeline._make_predict_timestamps(
            ts=example_tsds, start_timestamp=start_timestamp, end_timestamp=end_timestamp
        )
