import pandas as pd


def determine_num_steps_to_forecast(
    last_train_timestamp: pd.Timestamp, last_test_timestamp: pd.Timestamp, freq: str
) -> int:
    """Determine number of steps to make a forecast in future.

    It is useful for out-sample forecast with gap if model predicts only on a certain number of steps
    in autoregressive manner.

    Parameters
    ----------
    last_train_timestamp:
        last timestamp in train data
    last_test_timestamp:
        last timestamp in test data, should be after ``last_train_timestamp``
    freq:
        pandas frequency string: `Offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_

    Returns
    -------
    :
        number of steps

    Raises
    ------
    ValueError:
        Value of last test timestamp is less or equal than last train timestamp
    ValueError:
        Last train timestamp isn't correct according to a given frequency
    ValueError:
        Last test timestamps isn't reachable with a given frequency
    """
    if last_test_timestamp <= last_train_timestamp:
        raise ValueError("Last train timestamp should be less than last test timestamp!")

    # check if last_train_timestamp is normalized
    normalized_last_train_timestamp = pd.date_range(start=last_train_timestamp, periods=1, freq=freq)
    if normalized_last_train_timestamp != last_train_timestamp:
        raise ValueError(f"Last train timestamp isn't correct according to given frequency: {freq}")

    # make linear probing, because for complex offsets there is a cycle in `pd.date_range`
    cur_value = 1
    while True:
        timestamps = pd.date_range(start=last_train_timestamp, periods=cur_value + 1, freq=freq)
        if timestamps[-1] == last_test_timestamp:
            return cur_value
        elif timestamps[-1] > last_test_timestamp:
            raise ValueError(f"Last test timestamps isn't reachable with freq: {freq}")
        cur_value += 1
