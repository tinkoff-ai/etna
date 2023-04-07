import pandas as pd
import pytest

from etna.models.utils import determine_freq
from etna.models.utils import determine_num_steps
from etna.models.utils import select_observations


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


@pytest.fixture()
def df_without_timestamp():
    df = pd.DataFrame({"target": list(range(5))})
    return df


@pytest.mark.parametrize(
    "timestamps",
    (
        pd.to_datetime(pd.Series(["2020-02-01", "2020-02-03"])),
        pd.to_datetime(pd.Series(["2020-02-01"])),
    ),
)
def test_select_observations_without_timestamp(df_without_timestamp, timestamps):
    selected_df = select_observations(
        df=df_without_timestamp, timestamps=timestamps, freq="D", start="2020-02-01", periods=5
    )
    assert len(selected_df) == len(timestamps)


@pytest.mark.parametrize(
    "timestamps,answer",
    (
        (pd.date_range(start="2020-01-01", periods=3, freq="M"), "M"),
        (pd.date_range(start="2020-01-01", periods=3, freq="W"), "W-SUN"),
        (pd.date_range(start="2020-01-01", periods=3, freq="D"), "D"),
    ),
)
def test_determine_freq(timestamps, answer):
    assert determine_freq(timestamps=timestamps) == answer


@pytest.mark.parametrize(
    "timestamps",
    (
        pd.to_datetime(pd.Series(["2020-02-01", "2020-02-15", "2021-02-15"])),
        pd.to_datetime(pd.Series(["2020-02-15", "2020-01-22", "2020-01-23"])),
    ),
)
def test_determine_freq(timestamps):
    with pytest.raises(ValueError, match="Can't determine frequency of a given dataframe"):
        _ = determine_freq(timestamps=timestamps)
