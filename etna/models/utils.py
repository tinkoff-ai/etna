from typing import Literal
from typing import Union

import numpy as np
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
    while True:
        timestamps = pd.date_range(start=start_timestamp, periods=cur_value + 1, freq=freq)
        if timestamps[-1] == end_timestamp:
            return cur_value
        elif timestamps[-1] > end_timestamp:
            raise ValueError(f"End timestamp isn't reachable with freq: {freq}")
        cur_value += 1


def select_prediction_size_timestamps(
    prediction: Union[np.ndarray, pd.DataFrame], timestamp: pd.Series, prediction_size: int
) -> Union[np.ndarray, pd.DataFrame]:
    """Select last ``prediction_size`` timestamps in a given prediction.

    Parameters
    ----------
    prediction:
        prediction
    timestamp:
        timestamp series
    prediction_size
        number of last timestamps to select

    Returns
    -------
    :
        filtered prediction

    Raises
    ------
    ValueError:
        if value of ``prediction_size`` isn't positive
    ValueError:
        if value of ``prediction_size`` is bigger than number of timestamps
    """
    timestamp_unique = timestamp.unique()
    timestamp_unique.sort()
    if prediction_size <= 0:
        raise ValueError("Prediction size should be positive.")
    if prediction_size > len(timestamp_unique):
        raise ValueError("The value of prediction_size is bigger than number of timestamps, try to increase it.")
    border_value = timestamp_unique[-prediction_size]
    return prediction[timestamp >= border_value]


def check_prediction_size_value(
    prediction_size: Union[Literal["all"], int],
    num_timestamps: int,
    context_size: int,
) -> int:
    """Check prediction size value for a model.

    Parameters
    ----------
    prediction_size:
        prediction size for a model prediction
    num_timestamps:
        number of timestamps in a dataset given to prediction
    context_size
        context size of the model

    Returns
    -------
    :
        checked prediction_size

    Raises
    ------
    ValueError:
        if value of ``prediction_size`` isn't positive
    ValueError:
        if value of ``prediction_size`` is bigger than number of timestamps
    ValueError:
        if incorrect literal is used
    ValueError:
        if model requires context and ``prediction_size`` isn't set
    ValueError:
        context size is negative
    """
    if isinstance(prediction_size, int):
        if prediction_size <= 0:
            raise ValueError("Prediction size can be only positive!")
        elif prediction_size > num_timestamps:
            raise ValueError("Prediction size is bigger than number of timestamps!")
        else:
            return prediction_size
    else:
        if prediction_size != "all":
            raise ValueError("The only possible literal for prediction size is 'all'!")
        elif context_size > 0:
            raise ValueError("Literal 'all' is available only for model that has context_size equal to 0!")
        elif context_size < 0:
            raise ValueError("Context size can't be negative!")
        else:
            return num_timestamps
