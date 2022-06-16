import typing
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import List

import numba
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


@numba.jit(nopython=True)
def optimal_sse(left: int, right: int, p: np.ndarray, pp: np.ndarray) -> float:
    """
    Count the approximation error by 1 bin from left to right elements.

    Parameters
    ----------
    left:
        left border
    right:
        right border
    p:
        array of sums of elements, ``p[i]`` - sum from first to i elements
    pp:
        array of sums of squares of elements, ``pp[i]`` - sum of squares from first to i elements

    Returns
    -------
    result: float
        approximation error
    """
    if left == 0:
        avg = p[right]
        return pp[right] - avg**2 / (right - left + 1)
    avg = p[right] - p[left - 1]
    return pp[right] - pp[left - 1] - avg**2 / (right - left + 1)


@numba.jit(nopython=True)
def adjust_estimation(i: int, k: int, sse: np.ndarray, sse_one_bin: np.ndarray) -> float:
    """
    Count sse_one_bin[i][k] using binary search.

    Parameters
    ----------
    i:
        left border of series
    k:
        number of bins
    sse:
        array of approximation errors
    sse_one_bin:
        array of approximation errors with one bin

    Returns
    -------
    result: float
        calculated sse_one_bin[i][k]
    """
    now_evaluated = sse[i - 1][k - 1]
    first_evaluated = sse[i - 1][k - 1]
    idx_prev = np.inf
    idx_now = 0
    left = 0
    while idx_now != idx_prev:
        right = i
        idx_prev = idx_now
        while right - left > 1:
            if sse_one_bin[(left + right) // 2][i] > now_evaluated:
                left = (left + right) // 2
            else:
                right = (left + right) // 2
        idx_now = left
        now_evaluated = first_evaluated - sse[idx_now][k - 1]

    now_min = np.inf
    for j in range(idx_now, i):
        now = sse[j][k - 1] + sse_one_bin[j + 1][i]
        now_min = min(now_min, now)
    return now_min


@numba.jit(nopython=True)
def v_optimal_hist(series: np.ndarray, bins_number: int, p: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """
    Count an approximation error of a series with [1, bins_number] bins.

    `Reference <http://www.vldb.org/conf/1998/p275.pdf>`_.

    Parameters
    ----------
    series:
        array to count an approximation error with bins_number bins
    bins_number:
        number of bins
    p:
        array of sums of elements, p[i] - sum from 0th to i elements
    pp:
        array of sums of squares of elements, p[i] - sum of squares from 0th to i elements

    Returns
    -------
    error: np.ndarray
        approximation error of a series with [1, bins_number] bins
    """
    sse = np.zeros((len(series), bins_number))
    for i in range(len(series)):
        sse[i][0] = optimal_sse(0, i, p, pp)

    sse_one_bin = np.zeros((len(series), len(series)))
    for i in range(len(series)):
        for j in range(i, len(series)):
            sse_one_bin[i][j] = optimal_sse(i, j, p, pp)

    for tmp_bins_number in range(1, bins_number):
        for i in range(tmp_bins_number, len(series)):
            sse[i][tmp_bins_number] = adjust_estimation(i, tmp_bins_number, sse, sse_one_bin)
    return sse


def compute_f(series: np.ndarray, k: int, p: np.ndarray, pp: np.ndarray) -> typing.Tuple[np.ndarray, list]:
    """
    Compute F. F[a][b][k] - minimum approximation error on series[a:b+1] with k outliers.

    `Reference <http://www.vldb.org/conf/1999/P9.pdf>`_.

    Parameters
    ----------
    series:
        array to count F
    k:
        number of outliers
    p:
        array of sums of elements, ``p[i]`` - sum from 0th to i elements
    pp:
        array of sums of squares of elements, ``pp[i]`` - sum of squares from 0th to i elements

    Returns
    -------
    result: np.ndarray
        array F, outliers_indices
    """
    f = np.zeros((len(series), len(series), k + 1))
    s: list = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    ss: list = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    outliers_indices: list = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]

    for right_border in range(0, len(series)):
        f[0][right_border][0] = optimal_sse(0, right_border, p, pp)
        s[0][right_border][0] = [p[right_border]]
        ss[0][right_border][0] = [pp[right_border]]

    for left_border in range(1, len(series)):
        for right_border in range(left_border, len(series)):
            f[left_border][right_border][0] = optimal_sse(left_border, right_border, p, pp)
            s[left_border][right_border][0] = [p[right_border] - p[left_border - 1]]
            ss[left_border][right_border][0] = [pp[right_border] - pp[left_border - 1]]

    for left_border in range(0, len(series)):
        for right_border in range(left_border, min(len(series), left_border + k)):
            s[left_border][right_border][right_border - left_border + 1] = [0]
            ss[left_border][right_border][right_border - left_border + 1] = [0]
            outliers_indices[left_border][right_border][right_border - left_border + 1] = [
                list(np.arange(left_border, right_border + 1))
            ]

    for left_border in range(len(series)):
        for right_border in range(left_border + 1, len(series)):
            for outlier_number in range(1, min(right_border - left_border + 1, k + 1)):
                f1 = f[left_border][right_border - 1][outlier_number - 1]
                tmp_ss = []
                tmp_s = []
                f2 = []
                now_min = np.inf
                now_outliers_indices = []
                where = 0
                for i in range(len(ss[left_border][right_border - 1][outlier_number])):
                    tmp_ss.append(ss[left_border][right_border - 1][outlier_number][i] + series[right_border] ** 2)
                    tmp_s.append(s[left_border][right_border - 1][outlier_number][i] + series[right_border])
                    now_outliers_indices.append(
                        deepcopy(outliers_indices[left_border][right_border - 1][outlier_number][i])
                    )
                    f2.append(tmp_ss[-1] - tmp_s[-1] ** 2 / (right_border - left_border + 1 - outlier_number))
                    if f2[-1] < now_min:
                        now_min = f2[-1]
                        where = i

                if f1 < now_min:
                    f[left_border][right_border][outlier_number] = f1
                    s[left_border][right_border][outlier_number] = deepcopy(
                        s[left_border][right_border - 1][outlier_number - 1]
                    )
                    ss[left_border][right_border][outlier_number] = deepcopy(
                        ss[left_border][right_border - 1][outlier_number - 1]
                    )
                    outliers_indices[left_border][right_border][outlier_number] = deepcopy(
                        outliers_indices[left_border][right_border - 1][outlier_number - 1]
                    )
                    if len(outliers_indices[left_border][right_border][outlier_number]):
                        for i in range(len(outliers_indices[left_border][right_border][outlier_number])):
                            outliers_indices[left_border][right_border][outlier_number][i].append(right_border)
                    else:
                        outliers_indices[left_border][right_border][outlier_number].append([right_border])
                elif f1 > now_min:
                    f[left_border][right_border][outlier_number] = f2[where]
                    s[left_border][right_border][outlier_number] = tmp_s
                    ss[left_border][right_border][outlier_number] = tmp_ss

                    outliers_indices[left_border][right_border][outlier_number] = now_outliers_indices
                else:
                    f[left_border][right_border][outlier_number] = f1
                    tmp_s.extend(s[left_border][right_border - 1][outlier_number - 1])
                    tmp_ss.extend(ss[left_border][right_border - 1][outlier_number - 1])
                    s[left_border][right_border][outlier_number] = tmp_s
                    ss[left_border][right_border][outlier_number] = tmp_ss

                    tmp = deepcopy(outliers_indices[left_border][right_border - 1][outlier_number - 1])
                    if len(tmp):
                        for i in range(len(tmp)):
                            tmp[i].append(right_border)
                    else:
                        tmp = [[right_border]]
                    outliers_indices[left_border][right_border][outlier_number].extend(now_outliers_indices)
                    outliers_indices[left_border][right_border][outlier_number].extend(deepcopy(tmp))
    return f, outliers_indices


def hist(series: np.ndarray, bins_number: int) -> np.ndarray:
    """
    Compute outliers indices according to hist rule.

    `Reference <http://www.vldb.org/conf/1999/P9.pdf>`_.

    Parameters
    ----------
    series:
        array to count F
    bins_number:
        number of bins

    Returns
    -------
    indices: np.ndarray
        outliers indices
    """
    approximation_error = np.zeros((len(series), bins_number + 1, bins_number))
    anomalies: list = [[[[] for i in range(bins_number)] for j in range(bins_number + 1)] for s in range(len(series))]

    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2

    f, outliers_indices = compute_f(series, bins_number - 1, p, pp)

    approximation_error[:, 1:, 0] = v_optimal_hist(series, bins_number, p, pp)

    approximation_error[:, 1, :] = f[0]
    for right_border in range(len(series)):
        for outlier_number in range(1, bins_number):
            if len(outliers_indices[0][right_border][outlier_number]):
                anomalies[right_border][1][outlier_number] = deepcopy(
                    outliers_indices[0][right_border][outlier_number][0]
                )

    for right_border in range(1, len(series)):
        for tmp_bins_number in range(2, min(bins_number + 1, right_border + 2)):
            for outlier_number in range(1, min(bins_number, right_border + 2 - tmp_bins_number)):  # см формулу выше
                tmp_approximation_error = approximation_error[:right_border, tmp_bins_number - 1, : outlier_number + 1]
                tmp_f = f[1 : right_border + 1, right_border, : outlier_number + 1][:, ::-1]
                approximation_error[right_border][tmp_bins_number][outlier_number] = np.min(
                    tmp_approximation_error + tmp_f
                )
                where = np.where(
                    tmp_approximation_error + tmp_f
                    == approximation_error[right_border][tmp_bins_number][outlier_number]
                )

                if where[1][0] != outlier_number:
                    anomalies[right_border][tmp_bins_number][outlier_number].extend(
                        deepcopy(outliers_indices[1 + where[0][0]][right_border][outlier_number - where[1][0]][0])
                    )
                anomalies[right_border][tmp_bins_number][outlier_number].extend(
                    deepcopy(anomalies[where[0][0]][tmp_bins_number - 1][where[1][0]])
                )

    count = 0
    now_min = approximation_error[-1][-1][0]
    for outlier_number in range(1, min(approximation_error.shape[1], approximation_error.shape[2])):
        if approximation_error[-1][approximation_error.shape[1] - 1 - outlier_number][outlier_number] <= now_min:
            count = outlier_number
            now_min = approximation_error[-1][approximation_error.shape[1] - 1 - outlier_number][outlier_number]
    return np.array(sorted(anomalies[-1][approximation_error.shape[1] - 1 - count][count]))


def get_anomalies_hist(
    ts: "TSDataset", in_column: str = "target", bins_number: int = 10
) -> typing.Dict[str, List[pd.Timestamp]]:
    """
    Get point outliers in time series using histogram model.

    Outliers are all points that, when removed, result in a histogram with a lower approximation error,
    even with the number of bins less than the number of outliers.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    in_column:
        name of the column in which the anomaly is searching
    bins_number:
        number of bins

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}
    """
    outliers_per_segment = {}
    segments = ts.segments
    for seg in segments:
        segment_df = ts.df[seg].reset_index()
        values = segment_df[in_column].values
        timestamp = segment_df["timestamp"].values

        anomalies = hist(values, bins_number)

        outliers_per_segment[seg] = [timestamp[i] for i in anomalies]
    return outliers_per_segment
