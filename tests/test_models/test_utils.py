import pandas as pd
import pytest

from etna.models.utils import determine_num_steps_to_forecast


@pytest.mark.parametrize(
    "last_train_timestamp, last_test_timestamp, freq, answer",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), "D", 1),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11"), "D", 10),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-19"), "W-SUN", 2),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15"), pd.offsets.Week(), 2),
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2021-02-28"), "M", 13),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-01"), "MS", 17),
    ],
)
def test_determine_num_steps_to_forecast_ok(last_train_timestamp, last_test_timestamp, freq, answer):
    result = determine_num_steps_to_forecast(
        last_train_timestamp=last_train_timestamp, last_test_timestamp=last_test_timestamp, freq=freq
    )
    assert result == answer


@pytest.mark.parametrize(
    "last_train_timestamp, last_test_timestamp, freq",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), "D"),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01"), "D"),
    ],
)
def test_determine_num_steps_to_forecast_fail_wrong_order(last_train_timestamp, last_test_timestamp, freq):
    with pytest.raises(ValueError, match="Last train timestamp should be less than last test timestamp"):
        _ = determine_num_steps_to_forecast(
            last_train_timestamp=last_train_timestamp, last_test_timestamp=last_test_timestamp, freq=freq
        )


@pytest.mark.parametrize(
    "last_train_timestamp, last_test_timestamp, freq",
    [
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-01"), "M"),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-01"), "MS"),
    ],
)
def test_determine_num_steps_to_forecast_fail_wrong_start(last_train_timestamp, last_test_timestamp, freq):
    with pytest.raises(ValueError, match="Last train timestamp isn't correct according to given frequency"):
        _ = determine_num_steps_to_forecast(
            last_train_timestamp=last_train_timestamp, last_test_timestamp=last_test_timestamp, freq=freq
        )


@pytest.mark.parametrize(
    "last_train_timestamp, last_test_timestamp, freq",
    [
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-06-05"), "M"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-05"), "MS"),
    ],
)
def test_determine_num_steps_to_forecast_fail_wrong_end(last_train_timestamp, last_test_timestamp, freq):
    with pytest.raises(ValueError, match="Last test timestamps isn't reachable with freq"):
        _ = determine_num_steps_to_forecast(
            last_train_timestamp=last_train_timestamp, last_test_timestamp=last_test_timestamp, freq=freq
        )
