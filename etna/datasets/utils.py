from enum import Enum
from typing import Sequence

import pandas as pd

from etna.datasets.tsdataset import TSDataset


class DataFrameFormat(str, Enum):
    """Enum for different types of result."""

    wide = "wide"
    long = "long"


def duplicate_data(df: pd.DataFrame, segments: Sequence[str], format: str = DataFrameFormat.wide) -> pd.DataFrame:
    """Duplicate dataframe for all the segments.

    Parameters
    ----------
    df:
        dataframe to duplicate, there should be column "timestamp"
    segments:
        list of segments for making duplication
    format:
        represent the result in TSDataset inner format (wide) or in flatten format (long)

    Returns
    -------
    result: pd.DataFrame
        result of duplication for all the segments

    Raises
    ------
    ValueError:
        if segments list is empty
    ValueError:
        if incorrect strategy is given
    ValueError:
        if dataframe doesn't contain "timestamp" column

    Examples
    --------
    >>> from etna.datasets import generate_const_df
    >>> from etna.datasets import duplicate_data
    >>> from etna.datasets import TSDataset
    >>> df = generate_const_df(
    ...    periods=50, start_time="2020-03-10",
    ...    n_segments=2, scale=1
    ... )
    >>> timestamp = pd.date_range("2020-03-10", periods=100, freq="D")
    >>> is_friday_13 = (timestamp.weekday == 4) & (timestamp.day == 13)
    >>> df_exog_raw = pd.DataFrame({"timestamp": timestamp, "regressor_is_friday_13": is_friday_13})
    >>> df_exog = duplicate_data(df_exog_raw, segments=["segment_0", "segment_1"], format="wide")
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df=df_ts_format, df_exog=df_exog, freq="D")
    >>> ts.head()
    segment                 segment_0                     segment_1
    feature    regressor_is_friday_13 target regressor_is_friday_13 target
    timestamp
    2020-03-10                  False    1.0                  False    1.0
    2020-03-11                  False    1.0                  False    1.0
    2020-03-12                  False    1.0                  False    1.0
    2020-03-13                   True    1.0                   True    1.0
    2020-03-14                  False    1.0                  False    1.0
    """
    # check segments length
    if len(segments) == 0:
        raise ValueError("Parameter segments shouldn't be empty")

    # check format
    format_enum = DataFrameFormat(format)

    # check the columns
    if "timestamp" not in df.columns:
        raise ValueError("There should be 'timestamp' column")

    # construct long version
    segments_results = []
    for segment in segments:
        df_segment = df.copy()
        df_segment["segment"] = segment
        segments_results.append(df_segment)

    df_long = pd.concat(segments_results, ignore_index=True)

    # construct wide version if necessary
    if format_enum == DataFrameFormat.wide:
        df_wide = TSDataset.to_dataset(df_long)
        return df_wide

    return df_long
