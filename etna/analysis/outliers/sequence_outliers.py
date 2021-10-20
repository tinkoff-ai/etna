import warnings
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from saxpy.hotsax import find_discords_hotsax

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_segment_sequence_anomalies(
    series: np.ndarray, num_anomalies: int = 1, anomaly_length: int = 15, alphabet_size: int = 3, word_length: int = 3
) -> List[Tuple[int, int]]:
    """
    Get indices of start and end of sequence outliers for one segment using SAX HOT algorithm.

    Parameters
    ----------
    series:
        array to find outliers in
    num_anomalies:
        number of outliers to be found
    anomaly_length:
        target length of outliers
    alphabet_size:
        the number of letters with which the subsequence will be encrypted
    word_length:
        the number of segments into which the subsequence will be divided by the paa algorithm

    Returns
    -------
    list of tuples with start and end of outliers.
    """
    start_points = find_discords_hotsax(
        series=series, win_size=anomaly_length, num_discords=num_anomalies, a_size=alphabet_size, paa_size=word_length
    )

    result = [(pt[0], pt[0] + anomaly_length) for pt in start_points]

    return result


def get_sequence_anomalies(
    ts: "TSDataset",
    in_column: str = "target",
    num_anomalies: int = 1,
    anomaly_length: int = 15,
    alphabet_size: int = 3,
    word_length: int = 3,
) -> Dict[str, List[pd.Timestamp]]:
    """
    Find the start and end of the sequence outliers for each segment using the SAX HOT algorithm.
    We use saxpy under the hood.
    Repository link: https://github.com/seninp/saxpy.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    num_anomalies:
        number of outliers to be found
    anomaly_length:
        target length of outliers
    alphabet_size:
        the number of letters with which the subsequence will be encrypted
    word_length:
        the number of segments into which the subsequence will be divided by the paa algorithm

    Returns
    -------
    dict of outliers: Dict[str, List[pd.Timestamp]]
        dict of sequence outliers in format {segment: [outliers_timestamps]}
    """
    segments = ts.segments
    outliers_per_segment: Dict[str, list] = dict()

    for seg in segments:
        segment_df = ts[:, seg, :][seg]
        if segment_df[in_column].isnull().sum():
            warnings.warn(
                f"Segment {seg} contains nan-s. They will be removed when calculating outliers."
                + "Make sure this behavior is acceptable",
                RuntimeWarning,
            )
        segment_df = segment_df.dropna().reset_index()
        outliers_idxs = get_segment_sequence_anomalies(
            series=segment_df[in_column].values,
            num_anomalies=num_anomalies,
            anomaly_length=anomaly_length,
            alphabet_size=alphabet_size,
            word_length=word_length,
        )

        timestamps = segment_df["timestamp"].values
        outliers_per_segment[seg] = []
        for left_bound, right_bound in outliers_idxs:
            outliers_per_segment[seg].extend(timestamps[left_bound:right_bound])
    return outliers_per_segment


__all__ = ["get_sequence_anomalies"]
