from unittest.mock import MagicMock

import pytest

from etna.models import NaiveModel


def check_predict_fail_not_fitted(pipeline_constructor):
    pipeline = pipeline_constructor(model=NaiveModel(), horizon=1)
    with pytest.raises(ValueError, match="Pipeline is not fitted"):
        _ = pipeline.predict()


def check_predict_calls_validate_timestamps(pipeline_constructor, start_timestamp, end_timestamp, ts):
    pipeline = pipeline_constructor(model=NaiveModel(), horizon=1)
    pipeline.fit(ts)

    pipeline._validate_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    pipeline._validate_predict_timestamps.assert_called_once_with(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )


def check_predict_calls_validate_quantiles(pipeline_constructor, quantiles, ts):
    pipeline = pipeline_constructor(model=NaiveModel(), horizon=1)
    pipeline.fit(ts)

    pipeline._validate_predict_timestamps = MagicMock(return_value=(MagicMock(), MagicMock()))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    _ = pipeline.predict(quantiles=quantiles)

    pipeline._validate_quantiles.assert_called_once_with(quantiles=quantiles)


def check_predict_calls_private_predict(pipeline_constructor, prediction_interval, quantiles, ts):
    pipeline = pipeline_constructor(model=NaiveModel(), horizon=1)
    pipeline.fit(ts)

    start_timestamp_returned = MagicMock()
    end_timestamp_returned = MagicMock()
    pipeline._validate_predict_timestamps = MagicMock(return_value=(start_timestamp_returned, end_timestamp_returned))
    pipeline._validate_quantiles = MagicMock()
    pipeline._predict = MagicMock()

    result = pipeline.predict(prediction_interval=prediction_interval, quantiles=quantiles)

    pipeline._predict.assert_called_once_with(
        start_timestamp=start_timestamp_returned,
        end_timestamp=end_timestamp_returned,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
    )
    assert result == pipeline._predict.return_value
