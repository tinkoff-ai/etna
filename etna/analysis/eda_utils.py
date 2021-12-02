import math
import warnings
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Optional
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from statsmodels.graphics import utils

if TYPE_CHECKING:
    from etna.datasets import TSDataset

plot_acf = sm.graphics.tsa.plot_acf
plot_pacf = sm.graphics.tsa.plot_pacf


def cross_corr_plot(ts: "TSDataset", n_segments: int = 10, maxlags: int = 21, segments: Optional[Sequence] = None):
    """
    Cross-correlation plot between multiple timeseries.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    n_segments:
        number of random segments to plot
    maxlags:
        number of timeseries shifts for cross-correlation
    segments:
        segments to plot
    """
    if not segments:
        exist_segments = list(ts.segments)
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)
    segment_pairs = list(combinations(segments, r=2))
    if len(segment_pairs) == 0:
        raise ValueError("There are no pairs to plot! Try set n_segments > 1.")
    columns_num = min(2, len(segment_pairs))
    rows_num = math.ceil(len(segment_pairs) / columns_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True, squeeze=False)
    ax = ax.ravel()
    fig.suptitle("Cross-correlation", fontsize=16)
    for i, (segment_1, segment_2) in enumerate(segment_pairs):
        df_segment_1 = ts[:, segment_1, :][segment_1]
        df_segment_2 = ts[:, segment_2, :][segment_2]
        fig, axx = utils.create_mpl_ax(ax[i])
        target_1 = df_segment_1.target
        target_2 = df_segment_2.target
        if target_1.dtype == int or target_2.dtype == int:
            warnings.warn(
                "At least one target column has integer dtype, "
                "it is converted to float in order to calculate correlation."
            )
            target_1 = target_1.astype(float)
            target_2 = target_2.astype(float)
        lags, level, _, _ = axx.xcorr(x=target_1, y=target_2, maxlags=maxlags)
        ax[i].plot(lags, level, "o", markersize=5)
        ax[i].set_title(f"{segment_1} vs {segment_2}")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def sample_acf_plot(ts: "TSDataset", n_segments: int = 10, lags: int = 21, segments: Sequence = None):
    """
    Autocorrelation plot for multiple timeseries.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    n_segments:
        number of random segments to plot
    lags:
        number of timeseries shifts for cross-correlation
    segments:
        segments to plot

    Notes
    -----
    https://en.wikipedia.org/wiki/Autocorrelation
    """
    if not segments:
        segments = sorted(ts.segments)

    k = min(n_segments, len(segments))
    columns_num = min(2, k)
    rows_num = math.ceil(k / columns_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True, squeeze=False)
    ax = ax.ravel()
    fig.suptitle("Partial Autocorrelation", fontsize=16)
    for i, name in enumerate(sorted(np.random.choice(segments, size=k, replace=False))):
        df_slice = ts[:, name, :][name]
        plot_acf(x=df_slice["target"].values, ax=ax[i], lags=lags)
        ax[i].set_title(name)
    plt.show()


def sample_pacf_plot(ts: "TSDataset", n_segments: int = 10, lags: int = 21, segments: Sequence = None):
    """
    Partial autocorrelation plot for multiple timeseries.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    n_segments:
        number of random segments to plot
    lags:
        number of timeseries shifts for cross-correlation
    segments:
        segments to plot

    Notes
    -----
    https://en.wikipedia.org/wiki/Partial_autocorrelation_function
    """
    if not segments:
        segments = sorted(ts.segments)

    k = min(n_segments, len(segments))
    columns_num = min(2, k)
    rows_num = math.ceil(k / columns_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True, squeeze=False)
    ax = ax.ravel()
    fig.suptitle("Partial Autocorrelation", fontsize=16)
    for i, name in enumerate(sorted(np.random.choice(segments, size=k, replace=False))):
        df_slice = ts[:, name, :][name]
        plot_pacf(x=df_slice["target"].values, ax=ax[i], lags=lags)
        ax[i].set_title(name)
    plt.show()


def distribution_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    segments: Sequence = None,
    shift: int = 30,
    window: int = 30,
    freq: str = "1M",
    n_rows: int = 10,
):
    """Distribution of z-values grouped by segments and time frequency.

    ... math:
        mean_{i} = \\sum_{j=i-\\text{shift}}^{i-\\text{shift}+\\text{window}} \\frac{x_{j}}{\\text{window}}

    Parameters
    ----------
    ts:
        dataset with timeseries data
    n_segments:
        number of random segments to plot
    segments:
        segments to plot
    shift:
        number of timeseries shifts for statistics calc
    window:
        number of points for statistics calc
    freq:
        group for z_{i}
    n_rows:
        maximum number of rows to plot
    """
    df_pd = ts.to_pandas(flatten=True)

    if not segments:
        exist_segments = df_pd.segment.unique()
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)
    df_full = df_pd[df_pd.segment.isin(segments)]
    df_full.loc[:, "mean"] = (
        df_full.groupby("segment").target.shift(shift).transform(lambda s: s.rolling(window).mean())
    )
    df_full.loc[:, "std"] = df_full.groupby("segment").target.shift(shift).transform(lambda s: s.rolling(window).std())
    df_full = df_full.dropna()
    df_full.loc[:, "z"] = (df_full["target"] - df_full["mean"]) / df_full["std"]

    grouped_data = df_full.groupby([df_full.timestamp.dt.to_period(freq)])
    columns_num = min(2, len(grouped_data))
    rows_num = min(n_rows, math.ceil(len(grouped_data) / columns_num))
    groups = set(list(grouped_data.groups.keys())[-rows_num * columns_num :])
    fig, ax = plt.subplots(rows_num, columns_num, figsize=(20, 7.5 * rows_num), constrained_layout=True, squeeze=False)
    fig.suptitle(f"Z statistic shift: {shift} window: {window}", fontsize=16)
    ax = ax.ravel()
    i = 0
    for period, df_slice in grouped_data:
        if period not in groups:
            continue
        sns.boxplot(data=df_slice.sort_values(by="segment"), y="z", x="segment", ax=ax[i], fliersize=False)
        ax[i].set_title(f"{period}")
        i += 1
