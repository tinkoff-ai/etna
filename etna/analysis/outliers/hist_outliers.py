import typing
from typing import TYPE_CHECKING
from copy import deepcopy
from typing import List

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def SSE(i: int, j: int, p: List[float], pp: List[float]):
    """
    Count the approximation error by 1 bin from i to j elements.

    Parameters
    ----------
    i:
        left border
    j:
        right border
    p:
        array of sums of elements, p[i] - sum from first to i elements
    pp:
        array of sums of squares of elements, p[i] - sum of squares from first to i elements
    Returns
    -------
    Approximation error.
    """
    if i == 0:
        avg = p[j]
        return pp[j] - avg ** 2 / (j - i + 1)
    avg = p[j] - p[i - 1]
    return pp[j] - pp[i - 1] - avg ** 2 / (j - i + 1)


def v_optimal_hist(series: List[float], B: int):
    """
    Count an approximation error of a series with B bins.
    http://www.vldb.org/conf/1998/p275.pdf

    Parameters
    ----------
    series:
        array to count an approximation error with B bins
    B:
        number of bins
    Returns
    -------
    Approximation error of a series with B bins
    """
    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]  # p[i] = series[0] + series[1] + .. + series[i]
    pp[0] = series[0] ** 2  # pp[i] = series[0]**2 + series[1]**2 + .. + series[i]**2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2

    sse = np.zeros((len(series), B))  # sse[i][j] = ошибка аппроксимации j+1 бинами ряда series[:i+1]
    for i in range(len(series)):  # заполняем столбец матрицы для 1 бина
        sse[i][0] = SSE(0, i, p, pp)

    count_sse = np.zeros(
        (len(series), len(series))
    )  # count_sse[i][j] = ошибка аппроксимации с 1 бином от series[i:j+1]
    for i in range(len(series)):  # препдосчитываем для того чтобы тысячу раз не вызывать одно и то же
        for j in range(i, len(series)):
            count_sse[i][j] = SSE(i, j, p, pp)

    # начинаем заполнять sse
    for k in range(1, B):  # итерация по бинам
        for i in range(k, len(series)):  # итерация по ряду
            # заполняем sse[i][k]
            s1 = sse[i - 1][k - 1]
            s0 = sse[i - 1][k - 1]
            idx0 = np.inf
            idx1 = 0
            left = 0
            right = i
            while idx1 != idx0:
                right = i
                idx0 = idx1
                # найти бинпоиском такое j: count_sse[j][i] > s1
                while right - left > 1:
                    if count_sse[(left + right) // 2][i] > s1:
                        left = (left + right) // 2
                    else:
                        right = (left + right) // 2
                idx1 = left
                s1 = s0 - sse[idx1][k - 1]

            now_min = np.inf
            for j in range(idx1, i):
                now = sse[j][k - 1] + count_sse[j + 1][i]
                if now < now_min:
                    now_min = now
            sse[i][k] = now_min
    return sse[len(series) - 1][B - 1]


def computeF(series: List[float], k: int, p: List[float], pp: List[float]):
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
        array of sums of elements, p[i] - sum from first to i elements
    pp:
        array of sums of squares of elements, p[i] - sum of squares from first to i elements

    Returns
    -------
    Array F
    """
    F = np.zeros((len(series), len(series), k + 1))
    S = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    # S[i][j][k] - сумма всех элементов невыбросов с i по j, с учетом что там k выбросов
    SS = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    # SS[i][j][k] - сумма квадратов всех элементов невыбросов с i по j, с учетом что там k выбросов
    idx = [[[[] for i in range(k + 1)] for j in range(len(series))] for s in range(len(series))]
    # idx[i][j][k] - индексы выбросов (которые мы назвали такими про подсчете S и SS)
    # Везде сверху возможны несколько значений, поэтому листы. возникает когда F1 == F2 (ниже про них)

    # заполнение граничных условий
    for bi in range(0, len(series)):  # a = 0, c = 0
        F[0][bi][0] = SSE(0, bi, p, pp)
        S[0][bi][0] = [p[bi]]
        SS[0][bi][0] = [pp[bi]]

    for ai in range(1, len(series)):
        for bi in range(ai, len(series)):  # c = 0
            F[ai][bi][0] = SSE(ai, bi, p, pp)
            S[ai][bi][0] = [p[bi] - p[ai - 1]]
            SS[ai][bi][0] = [pp[bi] - pp[ai - 1]]

    for ai in range(len(series)):
        for bi in range(ai + 1, len(series)):
            for ci in range(1, min(bi - ai + 1, k + 1)):
                # рассматриваем новое значение bi. Если считаем его выбросом, то ошибка F1
                F1 = F[ai][bi - 1][ci - 1]
                # подсчет второго варианта, когда bi выбросом не является
                tmp_SS = []
                tmp_S = []
                F2 = []
                now_min = np.inf
                now_idx = []
                where = 0
                for i in range(
                    len(SS[ai][bi - 1][ci])
                ):  # формулы для пересчета коэффициентов и ошибок, если bi не выброс
                    tmp_SS.append(SS[ai][bi - 1][ci][i] + series[bi] ** 2)
                    tmp_S.append(S[ai][bi - 1][ci][i] + series[bi])
                    now_idx.append(deepcopy(idx[ai][bi - 1][ci][i]))
                    F2.append(tmp_SS[-1] - tmp_S[-1] ** 2 / (bi - ai + 1 - ci))
                    if F2[-1] < now_min:
                        now_min = F2[-1]
                        where = i

                if F1 < now_min:  # ошибка меньше в предположении bi выброс
                    F[ai][bi][ci] = F1
                    S[ai][bi][ci] = S[ai][bi - 1][ci - 1]
                    SS[ai][bi][ci] = SS[ai][bi - 1][ci - 1]
                    idx[ai][bi][ci] = deepcopy(idx[ai][bi - 1][ci - 1])
                    if len(idx[ai][bi][ci]):
                        for i in range(len(idx[ai][bi][ci])):
                            idx[ai][bi][ci][i].append(bi)
                    else:
                        idx[ai][bi][ci].append([bi])
                elif F1 > now_min:  # ошибка меньше в предположении bi не выброс
                    F[ai][bi][ci] = F2[where]
                    S[ai][bi][ci] = tmp_S
                    SS[ai][bi][ci] = tmp_SS

                    idx[ai][bi][ci] = now_idx
                else:  # плохой случай, когда в обоих случаях одинаково
                    # здесь не обязательно ВСЕ значения переписывать от S, SS и idx, можно только те, на которых ошибка минимальна
                    # но пока что так
                    F[ai][bi][ci] = F1
                    tmp_S.extend(S[ai][bi - 1][ci - 1])
                    tmp_SS.extend(SS[ai][bi - 1][ci - 1])
                    S[ai][bi][ci] = tmp_S
                    SS[ai][bi][ci] = tmp_SS

                    tmp = deepcopy(idx[ai][bi - 1][ci - 1])
                    if len(tmp):
                        for i in range(len(tmp)):
                            tmp[i].append(bi)
                    else:
                        tmp = [[bi]]
                    idx[ai][bi][ci].extend(now_idx)
                    idx[ai][bi][ci].extend(deepcopy(tmp))

    for ai in range(len(series)):  # просто проверка на соответствие длин, пусть пока что поживет тут
        for bi in range(ai + 1, len(series)):
            for ci in range(1, min(bi - ai + 1, k + 1)):
                for i in range(len(idx[ai][bi][ci])):
                    assert len(idx[ai][bi][ci][i]) == ci
    return F, idx


def hist(
    series: List[float], B: int
):  # главная функция, B - количество бинов, работает за N^2 * B^3, самое долгое - подсчет F
    """
    Compute outliers indices according to hist rule.
    http://www.vldb.org/conf/1999/P9.pdf

    Parameters
    ----------
    series:
        array to count F
    B:
        number of bins

    Returns
    -------
    Outliers indices.
    """
    # E[i][j][k] ошибка на series[:i+1] с j бинами и k выбросами
    # E[i][j][k] = min[1<= l <= i, 0 <= m <= k] (E[l, j-1, m] + F[l+1, i, k-m])
    E = np.zeros((len(series), B + 1, B))
    anomal = [
        [[[] for i in range(B)] for j in range(B + 1)] for s in range(len(series))
    ]  # храним индексы аномалий для E[i][j][k]

    p, pp = np.empty_like(series), np.empty_like(series)
    p[0] = series[0]
    pp[0] = series[0] ** 2
    for i in range(1, len(series)):
        p[i] = p[i - 1] + series[i]
        pp[i] = pp[i - 1] + series[i] ** 2

    F, idx = computeF(series, B - 1, p, pp)

    # граничные условия
    for i in range(len(series)):
        for j in range(1, B + 1):
            E[i][j][0] = v_optimal_hist(series[: i + 1], j)

    E[:, 1, :] = F[0]
    for i in range(len(series)):
        for k in range(1, B):
            if len(idx[0][i][k]):
                anomal[i][1][k] = deepcopy(idx[0][i][k][0])

    for i in range(1, len(series)):
        for j in range(2, min(B + 1, i + 2)):
            for k in range(1, min(B, i + 2 - j)):  # см формулу выше
                tmp_E = E[:i, j - 1, : k + 1]
                tmp_F = F[1 : i + 1, i, : k + 1][:, ::-1]
                E[i][j][k] = np.min(tmp_E + tmp_F)
                where = np.where(tmp_E + tmp_F == E[i][j][k])

                if where[1][0] != k:
                    anomal[i][j][k].extend(deepcopy(idx[1 + where[0][0]][i][k - where[1][0]][0]))
                anomal[i][j][k].extend(deepcopy(anomal[where[0][0]][j - 1][where[1][0]]))

                assert len(anomal[i][j][k]) == k  # тоже проверочка

    # берем минимальную ошибку от E[len(series)][B-i][i] по всем допустимым i, это i - количество выбросов
    count = 0
    now_min = E[-1][-1][0]
    for i in range(1, min(E.shape[1], E.shape[2])):
        if E[-1][E.shape[1] - 1 - i][i] <= now_min:
            count = i
            now_min = E[-1][E.shape[1] - 1 - i][i]
    return np.array(sorted(anomal[-1][E.shape[1] - 1 - count][count]))


def get_anomalies_hist(ts: "TSDataset", B: int = 10) -> typing.Dict[str, List[pd.Timestamp]]:
    """
    Get point outliers in time series using histogram model.
    Outliers are all points that, when removed, result in a histogram with a lower approximation error, even with the number of bins less than the number of outliers.
    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    B:
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

        anomal = hist(values, B)

        outliers_per_segment[seg] = [timestamp[i] for i in anomal]
    return outliers_per_segment
