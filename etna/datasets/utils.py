import re
from enum import Enum
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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


def match_target_quantiles(features: Set[str]) -> Set[str]:
    """Find quantiles in dataframe columns."""
    pattern = re.compile("target_\d+\.\d+$")
    return {i for i in list(features) if pattern.match(i) is not None}


def get_target_with_quantiles(columns: pd.Index) -> Set[str]:
    """Find "target" column and target quantiles among dataframe columns."""
    column_names = set(columns.get_level_values(level="feature"))
    target_columns = match_target_quantiles(column_names)
    if "target" in column_names:
        target_columns.add("target")
    return target_columns


def get_level_dataframe(
    df: pd.DataFrame,
    mapping_matrix: csr_matrix,
    source_level_segments: List[str],
    target_level_segments: List[str],
):
    """Perform mapping to dataframe at the target level.

    Parameters
    ----------
    df:
        dataframe at the source level
    mapping_matrix:
        mapping matrix between levels
    source_level_segments:
        tuple of segments at the source level
    target_level_segments:
        tuple of segments at the target level

    Returns
    -------
    :
       dataframe at the target level
    """
    target_names = tuple(get_target_with_quantiles(columns=df.columns))

    num_target_names = len(target_names)
    num_source_level_segments = len(source_level_segments)
    num_target_level_segments = len(target_level_segments)

    if len(target_names) == 0:
        raise ValueError("Provided dataframe has no columns with the target variable!")

    if set(df.columns.get_level_values(level="segment")) != set(source_level_segments):
        raise ValueError("Segments mismatch for provided dataframe and `source_level_segments`!")

    if num_source_level_segments != mapping_matrix.shape[1]:
        raise ValueError("Number of source level segments do not match mapping matrix number of columns!")

    if num_target_level_segments != mapping_matrix.shape[0]:
        raise ValueError("Number of target level segments do not match mapping matrix number of columns!")

    df = df.loc[:, pd.IndexSlice[source_level_segments, target_names]]

    source_level_data = df.values  # shape: (t, num_source_level_segments * num_target_names)

    source_level_data = source_level_data.reshape(
        (-1, num_source_level_segments, num_target_names)
    )  # shape: (t, num_source_level_segments, num_target_names)
    source_level_data = np.swapaxes(source_level_data, 1, 2)  # shape: (t, num_target_names, num_source_level_segments)
    source_level_data = source_level_data.reshape(
        (-1, num_source_level_segments)
    )  # shape: (t * num_target_names, num_source_level_segments)

    target_level_data = source_level_data @ mapping_matrix.T

    target_level_data = target_level_data.reshape(
        (-1, num_target_names, num_target_level_segments)
    )  # shape: (t, num_target_names, num_target_level_segments)
    target_level_data = np.swapaxes(target_level_data, 1, 2)  # shape: (t, num_target_level_segments, num_target_names)
    target_level_data = target_level_data.reshape(
        (-1, num_target_names * num_target_level_segments)
    )  # shape: (t, num_target_level_segments * num_target_names)

    target_level_segments = pd.MultiIndex.from_product(
        [target_level_segments, target_names], names=["segment", "feature"]
    )
    target_level_df = pd.DataFrame(data=target_level_data, index=df.index, columns=target_level_segments)

    return target_level_df
