from typing import Optional
from typing import Union

import pandas as pd


def determine_num_steps(start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, freq: str) -> int:
    """Determine how many steps of ``freq`` should we make from ``start_timestamp`` to reach ``end_timestamp``.

    Parameters
    ----------
    start_timestamp:
        timestamp to start counting from
    end_timestamp:
        timestamp to end counting, should be not earlier than ``start_timestamp``
    freq:
        pandas frequency string: `Offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_

    Returns
    -------
    :
        number of steps

    Raises
    ------
    ValueError:
        Value of end timestamp is less than start timestamp
    ValueError:
        Start timestamp isn't correct according to a given frequency
    ValueError:
        End timestamp isn't reachable with a given frequency
    """
    if start_timestamp > end_timestamp:
        raise ValueError("Start train timestamp should be less or equal than end timestamp!")

    # check if start_timestamp is normalized
    normalized_start_timestamp = pd.date_range(start=start_timestamp, periods=1, freq=freq)
    if normalized_start_timestamp != start_timestamp:
        raise ValueError(f"Start timestamp isn't correct according to given frequency: {freq}")

    # check a simple case
    if start_timestamp == end_timestamp:
        return 0

    # make linear probing, because for complex offsets there is a cycle in `pd.date_range`
    cur_value = 1
    cur_timestamp = start_timestamp
    while True:
        timestamps = pd.date_range(start=cur_timestamp, periods=2, freq=freq)
        if timestamps[-1] == end_timestamp:
            return cur_value
        elif timestamps[-1] > end_timestamp:
            raise ValueError(f"End timestamp isn't reachable with freq: {freq}")
        cur_value += 1
        cur_timestamp = timestamps[-1]


def select_observations(
    df: pd.DataFrame,
    timestamps: pd.Series,
    freq: str,
    start: Optional[Union[pd.Timestamp, str]] = None,
    end: Optional[Union[pd.Timestamp, str]] = None,
    periods: Optional[int] = None,
) -> pd.DataFrame:
    """Select observations from dataframe with known timeline.

    Parameters
    ----------
    df:
        dataframe with known timeline
    timestamps:
        series of timestamps to select
    freq:
        pandas frequency string
    start:
        start of the timeline
    end:
        end of the timeline
    periods:
        number of periods in the timeline

    Returns
    -------
    :
        dataframe with selected observations
    """
    df["timestamp"] = pd.date_range(start=start, end=end, periods=periods, freq=freq)

    if not (set(timestamps) <= set(df["timestamp"])):
        raise ValueError("Some timestamps do not lie inside the timeline of the provided dataframe.")

    observations = df.set_index("timestamp")
    observations = observations.loc[timestamps]
    observations.reset_index(drop=True, inplace=True)
    return observations


def determine_freq(timestamps: Union[pd.Series, pd.DatetimeIndex]) -> str:
    """Determine data frequency using provided timestamps.

    Parameters
    ----------
    timestamps:
        timeline to determine frequency

    Returns
    -------
    :
        pandas frequency string

    Raises
    ------
    ValueError:
        unable do determine frequency of data
    """
    try:
        freq = pd.infer_freq(timestamps, warn=False)
    except ValueError:
        freq = None

    if freq is None:
        raise ValueError("Can't determine frequency of a given dataframe")

    return freq
