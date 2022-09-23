from typing import Sequence
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.pipeline.base import BasePipeline


@pytest.fixture
def ts_with_different_beginnings(example_tsds):
    df = example_tsds.to_pandas()
    df.iloc[:5, 0] = np.NaN
    return TSDataset(df=df, freq="D")


class DummyPipeline(BasePipeline):
    def __init__(self, horizon: int):
        super().__init__(horizon=horizon)

    def fit(self, ts: TSDataset):
        self.ts = ts
        return self

    def _forecast(self):
        return None

    def _predict(
        self,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
        prediction_interval: bool,
        quantiles: Sequence[float],
    ) -> TSDataset:
        return self.ts


@pytest.mark.parametrize("quantiles", [(0.025,), (0.975,), (0.025, 0.975)])
@pytest.mark.parametrize("prediction_interval", [False, True])
@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")),
        (pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-15")),
    ],
)
@pytest.mark.parametrize(
    "ts", [TSDataset(df=TSDataset.to_dataset(generate_ar_df(start_time="2020-01-01", periods=5)), freq="D")]
)
def test_predict_pass_params(ts, start_timestamp, end_timestamp, prediction_interval, quantiles):
    pipeline = DummyPipeline(horizon=5)
    mock = MagicMock()
    pipeline._predict = mock

    pipeline.fit(ts)
    _ = pipeline.predict(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
    )

    mock.assert_called_once_with(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
    )


def test_predict_fail_not_fitted():
    pipeline = DummyPipeline(horizon=5)
    with pytest.raises(ValueError, match="DummyPipeline is not fitted"):
        _ = pipeline.predict()


@pytest.mark.parametrize("ts_name", ["example_tsds", "ts_with_different_beginnings"])
def test_predict_use_ts_timestamps(ts_name, request):
    ts = request.getfixturevalue(ts_name)
    pipeline = DummyPipeline(horizon=5)
    mock = MagicMock()
    pipeline._predict = mock

    pipeline.fit(ts)
    _ = pipeline.predict()

    expected_start_timestamp = ts.describe()["start_timestamp"].max()
    expected_end_timestamp = ts.index.max()

    mock.assert_called_once_with(
        start_timestamp=expected_start_timestamp,
        end_timestamp=expected_end_timestamp,
        prediction_interval=False,
        quantiles=(0.025, 0.975),
    )


def test_predict_fail_early_start(example_tsds):
    pipeline = DummyPipeline(horizon=5)
    pipeline.fit(example_tsds)
    start_timestamp = example_tsds.index[0] - pd.DateOffset(days=5)

    with pytest.raises(ValueError, match="Value of start_timestamp is less than beginning of some segments"):
        _ = pipeline.predict(start_timestamp=start_timestamp)


def test_predict_fail_late_end(example_tsds):
    pipeline = DummyPipeline(horizon=5)

    pipeline.fit(example_tsds)
    end_timestamp = example_tsds.index[-1] + pd.DateOffset(days=5)

    with pytest.raises(ValueError, match="Value of end_timestamp is more than ending of dataset"):
        _ = pipeline.predict(end_timestamp=end_timestamp)


def test_predict_fail_start_later_than_end(example_tsds):
    pipeline = DummyPipeline(horizon=5)

    pipeline.fit(example_tsds)
    start_timestamp = example_tsds.index[2]
    end_timestamp = example_tsds.index[0]

    with pytest.raises(ValueError, match="Value of end_timestamp is less than start_timestamp"):
        _ = pipeline.predict(start_timestamp=start_timestamp, end_timestamp=end_timestamp)


@pytest.mark.parametrize("quantiles", [(0.025,), (0.975,), (0.025, 0.975)])
def test_predict_validate_quantiles(quantiles, example_tsds):
    pipeline = DummyPipeline(horizon=5)
    mock = MagicMock()
    pipeline._validate_quantiles = mock

    pipeline.fit(example_tsds)
    _ = pipeline.predict(prediction_interval=True, quantiles=quantiles)

    mock.assert_called_once_with(quantiles=quantiles)


def test_predict_return_private_predict(example_tsds):
    pipeline = DummyPipeline(horizon=5)
    mock = MagicMock()
    pipeline._predict = mock
    expected_result = mock.return_value

    pipeline.fit(example_tsds)
    returned_result = pipeline.predict()

    assert returned_result == expected_result
