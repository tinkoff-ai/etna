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
        array of sums of elements, p[i] - sum from first to i elements
    pp:
        array of sums of squares of elements, p[i] - sum of squares from first to i elements
    Returns
    -------
    Approximation error.
    """
    if left == 0:
        avg = p[right]
        return pp[right] - avg ** 2 / (right - left + 1)
    avg = p[right] - p[left - 1]
    return pp[right] - pp[left - 1] - avg ** 2 / (right - left + 1)


@numba.jit(nopython=True)
def adjust_estimation(i: int, k: int, sse: np.ndarray, sse_one_bin: np.ndarray) -> float:
    """
    Count sse_one_bin[i][k] using binary search.
    """
    now_evaluated = sse[i - 1][k - 1]
    first_evalueted = sse[i - 1][k - 1]
    idx_prev = np.inf
    idx_now = 0
    left = 0
    while idx_now != idx_prev:
        right = i
        idx_prev = idx_now
        # найти бинпоиском такое j: sse_one_bin[j][i] > now_evaluated
        while right - left > 1:
            if sse_one_bin[(left + right) // 2][i] > now_evaluated:
                left = (left + right) // 2
            else:
                right = (left + right) // 2
        idx_now = left
        now_evaluated = first_evalueted - sse[idx_now][k - 1]

    now_min = np.inf
    for j in range(idx_now, i):
        now = sse[j][k - 1] + sse_one_bin[j + 1][i]
        now_min = min(now_min, now)
    return now_min


@numba.jit(nopython=True)
def v_optimal_hist(series: np.ndarray, bins_number: int, p: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """
    Count an approximation error of a series with [1, bins_number] bins.
    http://www.vldb.org/conf/1998/p275.pdf

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
    Approximation error of a series with [1, bins_number] bins
    """

    sse = np.zeros((len(series), bins_number))  # sse[i][j] = ошибка аппроксимации j+1 бинами ряда series[:i+1]
    for i in range(len(series)):  # заполняем столбец матрицы для 1 бина
        sse[i][0] = optimal_sse(0, i, p, pp)

    sse_one_bin = np.zeros(
        (len(series), len(series))
    )  # count_sse[i][j] = ошибка аппроксимации с 1 бином от series[i:j+1]
    for i in range(len(series)):  # препдосчитываем для того чтобы тысячу раз не вызывать одно и то же
        for j in range(i, len(series)):
            sse_one_bin[i][j] = optimal_sse(i, j, p, pp)

    # начинаем заполнять sse
    for tmp_bins_number in range(1, bins_number):  # итерация по бинам
        for i in range(tmp_bins_number, len(series)):  # итерация по ряду
            # заполняем sse[i][k]
            sse[i][tmp_bins_number] = adjust_estimation(i, tmp_bins_number, sse, sse_one_bin)
    return sse


def computeF(series: np.ndarray, k: int, p: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """
    Compute F. F[a][b][k] - minimum approximation error on series[a:b+1] with k outliers.
    http://www.vldb.org/conf/1999/P9.pdf

    Parameters
    ----------
    series:
        array to count F
    k:
        number of outliers
    p:
        array of sums of elements, p[i] - sum from 0th to i elements
    pp:
        array of sums of squares of elements, p[i] - sum of squares from 0th to i elements

    Returns
    -------
    Array F, outliers_indice
    """
    F = np.zeros((len(series), len(series), k + 1))
    S = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    # S[i][j][k] - сумма всех элементов невыбросов с i по j, с учетом что там k выбросов
    SS = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    # SS[i][j][k] - сумма квадратов всех элементов невыбросов с i по j, с учетом что там k выбросов
    outliers_indices = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    # idx[i][j][k] - индексы выбросов (которые мы назвали такими про подсчете S и SS)
    # Везде сверху возможны несколько значений, поэтому листы. возникает когда F1 == F2 (ниже про них)

    # заполнение граничных условий
    for right_border in range(0, len(series)):  # a = 0, c = 0
        F[0][right_border][0] = optimal_sse(0, right_border, p, pp)
        S[0][right_border][0] = [p[right_border]]
        SS[0][right_border][0] = [pp[right_border]]

    for left_border in range(1, len(series)):
        for right_border in range(left_border, len(series)):  # c = 0
            F[left_border][right_border][0] = optimal_sse(left_border, right_border, p, pp)
            S[left_border][right_border][0] = [p[right_border] - p[left_border - 1]]
            SS[left_border][right_border][0] = [pp[right_border] - pp[left_border - 1]]

    for left_border in range(0, len(series)):
        for right_border in range(left_border, min(len(series), left_border + k)):
            S[left_border][right_border][right_border - left_border + 1] = [0]
            SS[left_border][right_border][right_border - left_border + 1] = [0]
            outliers_indices[left_border][right_border][right_border - left_border + 1] = [
                list(np.arange(left_border, right_border + 1))
            ]

    for left_border in range(len(series)):
        for right_border in range(left_border + 1, len(series)):
            for outlier_number in range(1, min(right_border - left_border + 1, k + 1)):
                # рассматриваем новое значение bi. Если считаем его выбросом, то ошибка F1
                F1 = F[left_border][right_border - 1][outlier_number - 1]
                # подсчет второго варианта, когда bi выбросом не является
                tmp_SS = []
                tmp_S = []
                F2 = []
                now_min = np.inf
                now_outliers_indices = []
                where = 0
                for i in range(
                    len(SS[left_border][right_border - 1][outlier_number])
                ):  # формулы для пересчета коэффициентов и ошибок, если bi не выброс
                    tmp_SS.append(SS[left_border][right_border - 1][outlier_number][i] + series[right_border] ** 2)
                    tmp_S.append(S[left_border][right_border - 1][outlier_number][i] + series[right_border])
                    now_outliers_indices.append(
                        deepcopy(outliers_indices[left_border][right_border - 1][outlier_number][i])
                    )
                    F2.append(tmp_SS[-1] - tmp_S[-1] ** 2 / (right_border - left_border + 1 - outlier_number))
                    if F2[-1] < now_min:
                        now_min = F2[-1]
                        where = i

                if F1 < now_min:  # ошибка меньше в предположении bi выброс
                    F[left_border][right_border][outlier_number] = F1
                    S[left_border][right_border][outlier_number] = deepcopy(
                        S[left_border][right_border - 1][outlier_number - 1]
                    )
                    SS[left_border][right_border][outlier_number] = deepcopy(
                        SS[left_border][right_border - 1][outlier_number - 1]
                    )
                    outliers_indices[left_border][right_border][outlier_number] = deepcopy(
                        outliers_indices[left_border][right_border - 1][outlier_number - 1]
                    )
                    if len(outliers_indices[left_border][right_border][outlier_number]):
                        for i in range(len(outliers_indices[left_border][right_border][outlier_number])):
                            outliers_indices[left_border][right_border][outlier_number][i].append(right_border)
                    else:
                        outliers_indices[left_border][right_border][outlier_number].append([right_border])
                elif F1 > now_min:  # ошибка меньше в предположении bi не выброс
                    F[left_border][right_border][outlier_number] = F2[where]
                    S[left_border][right_border][outlier_number] = tmp_S
                    SS[left_border][right_border][outlier_number] = tmp_SS

                    outliers_indices[left_border][right_border][outlier_number] = now_outliers_indices
                else:  # плохой случай, когда в обоих случаях одинаково
                    # здесь не обязательно ВСЕ значения переписывать от S, SS и idx, можно только те, на которых ошибка минимальна
                    # но пока что так
                    F[left_border][right_border][outlier_number] = F1
                    tmp_S.extend(S[left_border][right_border - 1][outlier_number - 1])
                    tmp_SS.extend(SS[left_border][right_border - 1][outlier_number - 1])
                    S[left_border][right_border][outlier_number] = tmp_S
                    SS[left_border][right_border][outlier_number] = tmp_SS

                    tmp = deepcopy(outliers_indices[left_border][right_border - 1][outlier_number - 1])
                    if len(tmp):
                        for i in range(len(tmp)):
                            tmp[i].append(right_border)
                    else:
                        tmp = [[right_border]]
                    outliers_indices[left_border][right_border][outlier_number].extend(now_outliers_indices)
                    outliers_indices[left_border][right_border][outlier_number].extend(deepcopy(tmp))
    return F, outliers_indices


def hist(
    series: np.ndarray, bins_number: int
) -> np.ndarray:  # главная функция, bins_number - количество бинов, работает за N^2 * bins_number^3, самое долгое - подсчет F
    """
    Compute outliers indices according to hist rule.
    http://www.vldb.org/conf/1999/P9.pdf

    Parameters
    ----------
    series:
        array to count F
    bins_number:
        number of bins

    Returns
    -------
    Outliers indices.
    """
    # approximation_error[i][j][k] ошибка на series[:i+1] с j бинами и k выбросами
    # approximation_error[i][j][k] = min[1<= l <= i, 0 <= m <= k] (E[l, j-1, m] + F[l+1, i, k-m])
    approximation_error = np.zeros((len(series), bins_number + 1, bins_number))
    anomal = [
        [[[] for i in range(bins_number)] for j in range(bins_number + 1)] for s in range(len(series))
    ]  # храним индексы аномалий для E[i][j][k]

    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2

    F, idx = computeF(series, bins_number - 1, p, pp)

    # граничные условия
    approximation_error[:, 1:, 0] = v_optimal_hist(series, bins_number, p, pp)

    approximation_error[:, 1, :] = F[0]
    for right_border in range(len(series)):
        for outlier_number in range(1, bins_number):
            if len(idx[0][right_border][outlier_number]):
                anomal[right_border][1][outlier_number] = deepcopy(idx[0][right_border][outlier_number][0])

    for right_border in range(1, len(series)):
        for tmp_bins_number in range(2, min(bins_number + 1, right_border + 2)):
            for outlier_number in range(1, min(bins_number, right_border + 2 - tmp_bins_number)):  # см формулу выше
                tmp_approximation_error = approximation_error[:right_border, tmp_bins_number - 1, : outlier_number + 1]
                tmp_F = F[1 : right_border + 1, right_border, : outlier_number + 1][:, ::-1]
                approximation_error[right_border][tmp_bins_number][outlier_number] = np.min(
                    tmp_approximation_error + tmp_F
                )
                where = np.where(
                    tmp_approximation_error + tmp_F
                    == approximation_error[right_border][tmp_bins_number][outlier_number]
                )

                if where[1][0] != outlier_number:
                    anomal[right_border][tmp_bins_number][outlier_number].extend(
                        deepcopy(idx[1 + where[0][0]][right_border][outlier_number - where[1][0]][0])
                    )
                anomal[right_border][tmp_bins_number][outlier_number].extend(
                    deepcopy(anomal[where[0][0]][tmp_bins_number - 1][where[1][0]])
                )

    # берем минимальную ошибку от E[len(series)][bins_number-i][i] по всем допустимым i, это i - количество выбросов
    count = 0
    now_min = approximation_error[-1][-1][0]
    for outlier_number in range(1, min(approximation_error.shape[1], approximation_error.shape[2])):
        if approximation_error[-1][approximation_error.shape[1] - 1 - outlier_number][outlier_number] <= now_min:
            count = outlier_number
            now_min = approximation_error[-1][approximation_error.shape[1] - 1 - outlier_number][outlier_number]
    return np.array(sorted(anomal[-1][approximation_error.shape[1] - 1 - count][count]))


def get_anomalies_hist(ts: "TSDataset", bins_number: int = 10) -> typing.Dict[str, List[pd.Timestamp]]:
    """
    Get point outliers in time series using histogram model.
    Outliers are all points that, when removed, result in a histogram with a lower approximation error, even with the number of bins less than the number of outliers.
    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    bins_number:
        number of bins
    Returns
    -------
    dict of outliers: typing.Dict[str, typing.List[pd.Timestamp]]
        dict of outliers in format {segment: [outliers_timestamps]}
    """
    outliers_per_segment = {}
    segments = ts.segments
    for seg in segments:
        segment_df = ts.df[seg].reset_index()
        values = segment_df["target"].values
        timestamp = segment_df["timestamp"].values

        anomal = hist(values, bins_number)

        outliers_per_segment[seg] = [timestamp[i] for i in anomal]
    return outliers_per_segment
