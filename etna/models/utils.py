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
