import numpy as np
import pandas as pd
import pytest

from etna.datasets import generate_ar_df
from etna.models.utils import check_prediction_size_value
from etna.models.utils import determine_num_steps
from etna.models.utils import select_prediction_size_timestamps


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq, answer",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), "D", 1),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11"), "D", 10),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), "D", 0),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-19"), "W-SUN", 2),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15"), pd.offsets.Week(), 2),
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2021-02-28"), "M", 13),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-01"), "MS", 17),
    ],
)
def test_determine_num_steps_ok(start_timestamp, end_timestamp, freq, answer):
    result = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)
    assert result == answer


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01"), "D"),
    ],
)
def test_determine_num_steps_fail_wrong_order(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="Start train timestamp should be less or equal than end timestamp"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-01"), "M"),
        (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-06-01"), "MS"),
    ],
)
def test_determine_num_steps_fail_wrong_start(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="Start timestamp isn't correct according to given frequency"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, freq",
    [
        (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-06-05"), "M"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-05"), "MS"),
    ],
)
def test_determine_num_steps_fail_wrong_end(start_timestamp, end_timestamp, freq):
    with pytest.raises(ValueError, match="End timestamp isn't reachable with freq"):
        _ = determine_num_steps(start_timestamp=start_timestamp, end_timestamp=end_timestamp, freq=freq)


@pytest.mark.parametrize("prediction_size", [1, 5, 10])
def test_select_prediction_size_timestamps_df_ok(prediction_size):
    df = generate_ar_df(periods=10, start_time="2020-01-01", n_segments=3)
    df_selected = select_prediction_size_timestamps(
        prediction=df, timestamp=df["timestamp"], prediction_size=prediction_size
    )

    expected_timestamp = df["timestamp"].sort_values().unique()[-prediction_size:]
    df_expected = df[df["timestamp"].isin(expected_timestamp)]
    pd.testing.assert_frame_equal(df_selected, df_expected)


@pytest.mark.parametrize("prediction_size", [1, 5, 10])
def test_select_prediction_size_timestamps_array_ok(prediction_size):
    df = generate_ar_df(periods=10, start_time="2020-01-01", n_segments=3)
    array_selected = select_prediction_size_timestamps(
        prediction=df["target"].values, timestamp=df["timestamp"], prediction_size=prediction_size
    )

    expected_timestamp = df["timestamp"].sort_values().unique()[-prediction_size:]
    array_expected = df[df["timestamp"].isin(expected_timestamp)]["target"].values
    np.testing.assert_array_equal(array_selected, array_expected)


@pytest.mark.parametrize("prediction_size", [-1, 0])
def test_select_prediction_size_timestamps_fail_non_positive(prediction_size):
    df = generate_ar_df(periods=10, start_time="2020-01-01", n_segments=3)

    with pytest.raises(ValueError, match="Prediction size should be positive"):
        _ = select_prediction_size_timestamps(
            prediction=df["target"].values, timestamp=df["timestamp"], prediction_size=prediction_size
        )


def test_select_prediction_size_timestamps_fail_too_large():
    df = generate_ar_df(periods=10, start_time="2020-01-01", n_segments=3)

    with pytest.raises(ValueError, match="The value of prediction_size is bigger than number of timestamps"):
        _ = select_prediction_size_timestamps(
            prediction=df["target"].values, timestamp=df["timestamp"], prediction_size=11
        )


@pytest.mark.parametrize(
    "prediction_size, context_size, expected_value",
    [
        (5, 0, 5),
        (10, 0, 10),
        ("all", 0, 10),
    ],
)
def test_check_prediction_size_value_ok(prediction_size, context_size, expected_value):
    obtained_value = check_prediction_size_value(
        prediction_size=prediction_size, num_timestamps=10, context_size=context_size
    )
    assert obtained_value == expected_value


@pytest.mark.parametrize("prediction_size", [0, -1])
def test_check_prediction_size_value_fail_non_positive(prediction_size):
    with pytest.raises(ValueError, match="Prediction size can be only positive"):
        check_prediction_size_value(prediction_size=prediction_size, num_timestamps=10, context_size=0)


def test_check_prediction_size_value_fail_too_big():
    with pytest.raises(ValueError, match="Prediction size is bigger than number of timestamps"):
        check_prediction_size_value(prediction_size=11, num_timestamps=10, context_size=0)


def test_check_prediction_size_value_fail_wrong_literal():
    with pytest.raises(ValueError, match="The only possible literal for prediction size is 'all'"):
        check_prediction_size_value(prediction_size="random", num_timestamps=10, context_size=0)


def test_check_prediction_size_value_fail_required_context():
    with pytest.raises(ValueError, match="Literal 'all' is available only for model that has context_size equal to 0"):
        check_prediction_size_value(prediction_size="all", num_timestamps=10, context_size=1)


def test_check_prediction_size_value_fail_negative_context():
    with pytest.raises(ValueError, match="Context size can't be negative"):
        check_prediction_size_value(prediction_size="all", num_timestamps=10, context_size=-1)
