from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def absolute_difference_distance(x: float, y: float) -> float:
    """Calculate distance for :py:func:`get_anomalies_density` function by taking absolute value of difference.

    Parameters
    ----------
    x:
        first value
    y:
        second value

    Returns
    -------
    result: float
        absolute difference between values
    """
    return abs(x - y)


def get_segment_density_outliers_indices(
    series: np.ndarray,
    window_size: int = 7,
    distance_threshold: float = 10,
    n_neighbors: int = 3,
    distance_func: Callable[[float, float], float] = absolute_difference_distance,
) -> List[int]:
    """Get indices of outliers for one series.

    Parameters
    ----------
    series:
        array to find outliers in
    window_size:
        size of window
    distance_threshold:
        if distance between two items in the window is less than threshold those items are supposed to be close to each other
    n_neighbors:
        min number of close items that item should have not to be outlier
    distance_func:
        distance function

    Returns
    -------
    :
        list of outliers' indices
    """

    def is_close(item1: float, item2: float) -> int:
        """Return 1 if item1 is closer to item2 than distance_threshold according to distance_func, 0 otherwise."""
        return int(distance_func(item1, item2) < distance_threshold)

    outliers_indices = []
    for idx, item in enumerate(series):
        is_outlier = True
        left_start = max(0, idx - window_size)
        left_stop = max(0, min(idx, len(series) - window_size))
        closeness = None
        n = 0
        for i in range(left_start, left_stop + 1):
            if closeness is None:
                closeness = [is_close(item, series[j]) for j in range(i, min(i + window_size, len(series)))]
                n = sum(closeness) - 1
            else:
                n -= closeness.pop(0)
                new_element_is_close = is_close(item, series[i + window_size - 1])
                closeness.append(new_element_is_close)
                n += new_element_is_close
            if n >= n_neighbors:
                is_outlier = False
                break
        if is_outlier:
            outliers_indices.append(idx)
    return list(outliers_indices)


def get_anomalies_density(
    ts: "TSDataset",
    in_column: str = "target",
    window_size: int = 15,
    distance_coef: float = 3,
    n_neighbors: int = 3,
    distance_func: Callable[[float, float], float] = absolute_difference_distance,
) -> Dict[str, List[pd.Timestamp]]:
    """Compute outliers according to density rule.

    For each element in the series build all the windows of size ``window_size`` containing this point.
    If any of the windows contains at least ``n_neighbors`` that are closer than ``distance_coef * std(series)``
    to target point according to ``distance_func`` target point is not an outlier.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    window_size:
        size of windows to build
    distance_coef:
        factor for standard deviation that forms distance threshold to determine points are close to each other
    n_neighbors:
        min number of close neighbors of point not to be outlier
    distance_func:
        distance function

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}

    Notes
    -----
    It is a variation of distance-based (index) outlier detection method adopted for timeseries.
    """
    segments = ts.segments
    outliers_per_segment = {}
    for seg in segments:
        # TODO: dropna() now is responsible for removing nan-s at the end of the sequence and in the middle of it
        #   May be error or warning should be raised in this case
        segment_df = ts[:, seg, :][seg].dropna().reset_index()
        series = segment_df[in_column].values
        timestamps = segment_df["timestamp"].values
        series_std = np.std(series)
        if series_std:
            outliers_idxs = get_segment_density_outliers_indices(
                series=series,
                window_size=window_size,
                distance_threshold=distance_coef * series_std,
                n_neighbors=n_neighbors,
                distance_func=distance_func,
            )
            outliers = [timestamps[i] for i in outliers_idxs]
            outliers_per_segment[seg] = outliers
        else:
            outliers_per_segment[seg] = []
    return outliers_per_segment


__all__ = ["get_anomalies_density", "absolute_difference_distance"]
