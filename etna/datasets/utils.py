from enum import Enum
from typing import List
from typing import Optional
from typing import Sequence

import pandas as pd

from etna import SETTINGS

if SETTINGS.torch_required:
    from torch.utils.data import Dataset
else:
    from unittest.mock import Mock

    Dataset = Mock  # type: ignore


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
    >>> df_exog_raw = pd.DataFrame({"timestamp": timestamp, "is_friday_13": is_friday_13})
    >>> df_exog = duplicate_data(df_exog_raw, segments=["segment_0", "segment_1"], format="wide")
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df=df_ts_format, df_exog=df_exog, freq="D", known_future="all")
    >>> ts.head()
    segment       segment_0           segment_1
    feature    is_friday_13 target is_friday_13 target
    timestamp
    2020-03-10        False   1.00        False   1.00
    2020-03-11        False   1.00        False   1.00
    2020-03-12        False   1.00        False   1.00
    2020-03-13         True   1.00         True   1.00
    2020-03-14        False   1.00        False   1.00
    """
    from etna.datasets.tsdataset import TSDataset

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


class _TorchDataset(Dataset):
    """In memory dataset for torch dataloader."""

    def __init__(self, ts_samples: List[dict]):
        """Init torch dataset.

        Parameters
        ----------
        ts_samples:
            time series samples for training or inference
        """
        self.ts_samples = ts_samples

    def __getitem__(self, index):
        return self.ts_samples[index]

    def __len__(self):
        return len(self.ts_samples)


def set_columns_wide(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    timestamps_left: Optional[Sequence[pd.Timestamp]] = None,
    timestamps_right: Optional[Sequence[pd.Timestamp]] = None,
    segments_left: Optional[Sequence[str]] = None,
    features_right: Optional[Sequence[str]] = None,
    features_left: Optional[Sequence[str]] = None,
    segments_right: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Set columns in a left dataframe with values from the right dataframe.

    Parameters
    ----------
    df_left:
        dataframe to set columns in
    df_right:
        dataframe to set columns from
    timestamps_left:
        timestamps to select in ``df_left``
    timestamps_right:
        timestamps to select in ``df_right``
    segments_left:
        segments to select in ``df_left``
    segments_right:
        segments to select in ``df_right``
    features_left:
        features to select in ``df_left``
    features_right:
        features to select in ``df_right``

    Returns
    -------
    :
        a new dataframe with changed columns
    """
    # sort columns
    df_left = df_left.sort_index(axis=1)
    df_right = df_right.sort_index(axis=1)

    # prepare indexing
    timestamps_left_index = slice(None) if timestamps_left is None else timestamps_left
    timestamps_right_index = slice(None) if timestamps_right is None else timestamps_right
    segments_left_index = slice(None) if segments_left is None else segments_left
    segments_right_index = slice(None) if segments_right is None else segments_right
    features_left_index = slice(None) if features_left is None else features_left
    features_right_index = slice(None) if features_right is None else features_right

    right_value = df_right.loc[timestamps_right_index, (segments_right_index, features_right_index)]
    df_left.loc[timestamps_left_index, (segments_left_index, features_left_index)] = right_value.values

    return df_left
