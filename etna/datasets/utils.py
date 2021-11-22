from enum import Enum
from typing import Sequence

import pandas as pd

from etna.datasets.tsdataset import TSDataset


class DataFrameFormat(str, Enum):
    """Enum for different types of result."""

    wide = "wide"
    long = "long"


def duplicate_data(df: pd.DataFrame, segments: Sequence[str], format: str = DataFrameFormat.long) -> pd.DataFrame:
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
        if segments is empty
    ValueError:
        if incorrect strategy given
    ValueError:
        if dataframe doesn't contain "timestamp" column

    Examples
    --------
    # TODO: add example of usage
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
