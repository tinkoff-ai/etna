from unittest.mock import MagicMock

import pandas as pd
import pytest

from etna.datasets import TSDataset
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


class DummyPipeline(BasePipeline):
    def fit(self, ts: TSDataset):
        self.ts = ts
        return self

    def _forecast(self) -> TSDataset:
        return self.ts


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (None, None),
        (pd.Timestamp("2020-01-02"), None),
        (None, pd.Timestamp("2020-02-01")),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-02-01")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-02-03")),
    ],
)
def test_predict_calls_make_timestamps(start_timestamp, end_timestamp, example_tsds):
    pipeline = DummyPipeline(horizon=1)

    pipeline._make_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(ts=example_tsds, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    pipeline._make_predict_timestamps.assert_called_once_with(
        ts=example_tsds, start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )


@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,)])
def test_predict_calls_validate_quantiles(quantiles, example_tsds):
    pipeline = DummyPipeline(horizon=1)

    pipeline._make_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(ts=example_tsds, quantiles=quantiles)

    pipeline._validate_quantiles.assert_called_once_with(quantiles=quantiles)


@pytest.mark.parametrize("prediction_interval", [False, True])
@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,)])
def test_predict_calls_private_predict(prediction_interval, quantiles, example_tsds):
    pipeline = DummyPipeline(horizon=1)

    start_timestamp = MagicMock()
    end_timestamp = MagicMock()
    pipeline._make_predict_timestamps = MagicMock(return_value=(start_timestamp, end_timestamp))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(ts=example_tsds, prediction_interval=prediction_interval, quantiles=quantiles)

    pipeline._predict.assert_called_once_with(
        ts=example_tsds,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
    )
