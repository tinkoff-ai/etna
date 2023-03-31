import itertools
import math
import warnings
from copy import deepcopy
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.signal import periodogram
from typing_extensions import Literal

from etna.analysis.eda.utils import _create_holidays_df
from etna.analysis.eda.utils import get_correlation_matrix
from etna.analysis.feature_selection import AGGREGATION_FN
from etna.analysis.feature_selection import AggregationMode
from etna.analysis.utils import _get_borders_ts
from etna.analysis.utils import _prepare_axes

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.transforms import TimeSeriesImputerTransform

plot_acf = sm.graphics.tsa.plot_acf
plot_pacf = sm.graphics.tsa.plot_pacf


def plot_correlation_matrix(
    ts: "TSDataset",
    columns: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    method: str = "pearson",
    mode: str = "macro",
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 10),
    **heatmap_kwargs,
):
    """Plot pairwise correlation heatmap for selected segments.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    columns:
        Columns to use, if None use all columns
    segments:
        Segments to use
    method:
        Method of correlation:

        * pearson: standard correlation coefficient

        * kendall: Kendall Tau correlation coefficient

        * spearman: Spearman rank correlation

    mode: 'macro' or 'per-segment'
        Aggregation mode
    columns_num:
        Number of subplots columns
    figsize:
        size of the figure in inches
    """
    if segments is None:
        segments = sorted(ts.segments)
    if columns is None:
        columns = list(set(ts.df.columns.get_level_values("feature")))
    if "vmin" not in heatmap_kwargs:
        heatmap_kwargs["vmin"] = -1
    if "vmax" not in heatmap_kwargs:
        heatmap_kwargs["vmax"] = 1

    if mode not in ["macro", "per-segment"]:
        raise ValueError(f"'{mode}' is not a valid method of mode.")

    if mode == "macro":
        fig, ax = plt.subplots(figsize=figsize)
        correlation_matrix = get_correlation_matrix(ts, columns, segments, method)
        labels = list(ts[:, segments, columns].columns.values)
        ax = sns.heatmap(correlation_matrix, annot=True, fmt=".1g", square=True, ax=ax, **heatmap_kwargs)
        ax.set_xticks(np.arange(len(labels)) + 0.5, labels=labels)
        ax.set_yticks(np.arange(len(labels)) + 0.5, labels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
        ax.set_title("Correlation Heatmap")

    if mode == "per-segment":
        fig, ax = _prepare_axes(len(segments), columns_num=columns_num, figsize=figsize)

        for i, segment in enumerate(segments):
            correlation_matrix = get_correlation_matrix(ts, columns, [segment], method)
            labels = list(ts[:, segment, columns].columns.values)
            ax[i] = sns.heatmap(correlation_matrix, annot=True, fmt=".1g", square=True, ax=ax[i], **heatmap_kwargs)
            ax[i].set_xticks(np.arange(len(labels)) + 0.5, labels=labels)
            ax[i].set_yticks(np.arange(len(labels)) + 0.5, labels=labels)
            plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.setp(ax[i].get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
            ax[i].set_title("Correlation Heatmap" + " " + segment)


def plot_clusters(
    ts: "TSDataset",
    segment2cluster: Dict[str, int],
    centroids_df: Optional[pd.DataFrame] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot clusters [with centroids].

    Parameters
    ----------
    ts:
        TSDataset with timeseries
    segment2cluster:
        mapping from segment to cluster in format {segment: cluster}
    centroids_df:
        dataframe with centroids
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    unique_clusters = sorted(set(segment2cluster.values()))
    _, ax = _prepare_axes(num_plots=len(unique_clusters), columns_num=columns_num, figsize=figsize)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    segment_color = default_colors[0]
    for i, cluster in enumerate(unique_clusters):
        segments = [segment for segment in segment2cluster if segment2cluster[segment] == cluster]
        for segment in segments:
            segment_slice = ts[:, segment, "target"]
            ax[i].plot(
                segment_slice.index.values,
                segment_slice.values,
                alpha=1 / math.sqrt(len(segments)),
                c=segment_color,
            )
        ax[i].set_title(f"cluster={cluster}\n{len(segments)} segments in cluster")
        if centroids_df is not None:
            centroid = centroids_df[cluster, "target"]
            ax[i].plot(centroid.index.values, centroid.values, c="red", label="centroid")
        ax[i].legend()


def plot_periodogram(
    ts: "TSDataset",
    period: float,
    amplitude_aggregation_mode: Union[str, Literal["per-segment"]] = AggregationMode.mean,
    periodogram_params: Optional[Dict[str, Any]] = None,
    segments: Optional[List[str]] = None,
    xticks: Optional[List[Any]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot the periodogram using :py:func:`scipy.signal.periodogram`.

    It is useful to determine the optimal ``order`` parameter
    for :py:class:`~etna.transforms.timestamp.fourier.FourierTransform`.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    period:
        the period of the seasonality to capture in frequency units of time series, it should be >= 2;
        it is translated to the ``fs`` parameter of :py:func:`scipy.signal.periodogram`
    amplitude_aggregation_mode:
        aggregation strategy for obtained per segment periodograms;
        all the strategies can be examined
        at :py:class:`~etna.analysis.feature_selection.mrmr_selection.AggregationMode`
    periodogram_params:
        additional keyword arguments for periodogram, :py:func:`scipy.signal.periodogram` is used
    segments:
        segments to use
    xticks:
        list of tick locations of the x-axis, useful to highlight specific reference periodicities
    columns_num:
        if ``amplitude_aggregation_mode="per-segment"`` number of columns in subplots, otherwise the value is ignored
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if period < 2
    ValueError:
        if periodogram can't be calculated on segment because of the NaNs inside it

    Notes
    -----
    In non per-segment mode all segments are cut to be the same length, the last values are taken.
    """
    if period < 2:
        raise ValueError("Period should be at least 2")
    if periodogram_params is None:
        periodogram_params = {}
    if not segments:
        segments = sorted(ts.segments)

    df = ts.to_pandas()

    # plot periodograms
    if amplitude_aggregation_mode == "per-segment":
        _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
        for i, segment in enumerate(segments):
            segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
            segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()]
            if segment_df.isna().any():
                raise ValueError(f"Periodogram can't be calculated on segment with NaNs inside: {segment}")
            frequencies, spectrum = periodogram(x=segment_df, fs=period, **periodogram_params)
            spectrum = spectrum[frequencies >= 1]
            frequencies = frequencies[frequencies >= 1]
            ax[i].step(frequencies, spectrum)
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency")
            ax[i].set_ylabel("Power spectral density")
            if xticks is not None:
                ax[i].set_xticks(ticks=xticks, labels=xticks)
            ax[i].set_title(f"Periodogram: {segment}")
    else:
        # find length of each segment
        lengths_segments = []
        for segment in segments:
            segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
            segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()]
            if segment_df.isna().any():
                raise ValueError(f"Periodogram can't be calculated on segment with NaNs inside: {segment}")
            lengths_segments.append(len(segment_df))
        cut_length = min(lengths_segments)

        # cut each segment to `cut_length` last elements and find periodogram for each segment
        frequencies_segments = []
        spectrums_segments = []
        for segment in segments:
            segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
            segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()][-cut_length:]
            frequencies, spectrum = periodogram(x=segment_df, fs=period, **periodogram_params)
            frequencies_segments.append(frequencies)
            spectrums_segments.append(spectrum)

        frequencies = frequencies_segments[0]
        amplitude_aggregation_fn = AGGREGATION_FN[AggregationMode(amplitude_aggregation_mode)]
        spectrum = amplitude_aggregation_fn(spectrums_segments, axis=0)  # type: ignore
        spectrum = spectrum[frequencies >= 1]
        frequencies = frequencies[frequencies >= 1]
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.step(frequencies, spectrum)  # type: ignore
        ax.set_xscale("log")  # type: ignore
        ax.set_xlabel("Frequency")  # type: ignore
        ax.set_ylabel("Power spectral density")  # type: ignore
        if xticks is not None:
            ax.set_xticks(ticks=xticks, labels=xticks)  # type: ignore
        ax.set_title("Periodogram")  # type: ignore
        ax.grid()  # type: ignore


def plot_holidays(
    ts: "TSDataset",
    holidays: Union[str, pd.DataFrame],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    start: Optional[str] = None,
    end: Optional[str] = None,
    as_is: bool = False,
):
    """Plot holidays for segments.

    Sequence of timestamps with one holiday is drawn as a colored region.
    Individual holiday is drawn like a colored point.

    It is not possible to distinguish points plotted at one timestamp, but this case is considered rare.
    This the problem isn't relevant for region drawing because they are partially transparent.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    holidays:
        there are several options:

        * if str, then this is code of the country in `holidays <https://pypi.org/project/holidays/>`_ library;

        * if DataFrame, then dataframe is expected to be in prophet`s holiday format;

    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    as_is:
        * | Use this option if DataFrame is represented as a dataframe with a timestamp index and holiday names columns.
          | In a holiday column values 0 represent absence of holiday in that timestamp, 1 represent the presence.
    start:
        start timestamp for plot
    end:
        end timestamp for plot

    Raises
    ------
    ValueError:
        * Holiday nor pd.DataFrame or String.
        * Holiday is an empty pd.DataFrame.
        * `as_is=True` while holiday is String.
        * If upper_window is negative.
        * If lower_window is positive.

    """
    start, end = _get_borders_ts(ts, start, end)

    if segments is None:
        segments = sorted(ts.segments)

    holidays_df = _create_holidays_df(holidays, index=ts.index, as_is=as_is)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    df = ts.to_pandas()

    for i, segment in enumerate(segments):
        segment_df = df.loc[start:end, pd.IndexSlice[segment, "target"]]  # type: ignore
        segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()]

        # plot target on segment
        target_plot = ax[i].plot(segment_df.index, segment_df)
        target_color = target_plot[0].get_color()

        # plot holidays on segment
        # remember color of each holiday to reuse it
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        default_colors.remove(target_color)
        color_cycle = itertools.cycle(default_colors)
        holidays_colors = {holiday_name: next(color_cycle) for holiday_name in holidays_df.columns}

        for holiday_name in holidays_df.columns:
            holiday_df = holidays_df.loc[segment_df.index, holiday_name]
            for _, holiday_group in itertools.groupby(enumerate(holiday_df.tolist()), key=lambda x: x[1]):
                holiday_group_cached = list(holiday_group)
                indices = [x[0] for x in holiday_group_cached]
                values = [x[1] for x in holiday_group_cached]

                # if we have group with zero value, then it is not a holidays, skip it
                if values[0] == 0:
                    continue

                color = holidays_colors[holiday_name]
                if len(indices) == 1:
                    # plot individual holiday point
                    ax[i].scatter(segment_df.index[indices[0]], segment_df.iloc[indices[0]], color=color, zorder=2)
                else:
                    # plot span with holiday borders
                    x_min = segment_df.index[indices[0]]
                    x_max = segment_df.index[indices[-1]]
                    ax[i].axvline(x_min, color=color, linestyle="dashed")
                    ax[i].axvline(x_max, color=color, linestyle="dashed")
                    ax[i].axvspan(xmin=x_min, xmax=x_max, alpha=1 / 4, color=color)

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)

        legend_handles = [
            Line2D([0], [0], marker="o", color=color, label=label) for label, color in holidays_colors.items()
        ]
        ax[i].legend(handles=legend_handles)


def _cross_correlation(
    a: np.ndarray, b: np.ndarray, maxlags: Optional[int] = None, normed: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate cross correlation between arrays.

    This implementation is slow: O(n^2), but can properly ignore NaNs.

    Parameters
    ----------
    a:
        first array, should be equal length with b
    b:
        second array, should be equal length with a
    maxlags:
        number of lags to compare, should be >=1 and < len(a)
    normed:
        should correlations be normed or not

    Returns
    -------
    lags, result:

        * lags: array of size ``maxlags * 2 + 1`` represents for which lags correlations are calculated in ``result``

        * result: array of size ``maxlags * 2 + 1`` represents found correlations

    Raises
    ------
    ValueError:
        lengths of ``a`` and ``b`` are not the same
    ValueError:
        parameter ``maxlags`` doesn't satisfy constraints
    """
    if len(a) != len(b):
        raise ValueError("Lengths of arrays should be equal")

    length = len(a)

    if maxlags is None:
        maxlags = length - 1

    if maxlags < 1 or maxlags >= length:
        raise ValueError("Parameter maxlags should be >= 1 and < len(a)")

    result = []
    lags = np.arange(-maxlags, maxlags + 1)
    for lag in lags:
        if lag < 0:
            cur_a = a[:lag]
            cur_b = b[-lag:]
        elif lag == 0:
            cur_a = a
            cur_b = b
        else:
            cur_a = a[lag:]
            cur_b = b[:-lag]
        dot_product = np.nansum(cur_a * cur_b)
        if normed:
            nan_mask_a = np.isnan(cur_a)
            nan_mask_b = np.isnan(cur_b)
            nan_mask = nan_mask_a | nan_mask_b
            normed_dot_product = dot_product / np.sqrt(
                np.sum(cur_a[~nan_mask] * cur_a[~nan_mask]) * np.sum(cur_b[~nan_mask] * cur_b[~nan_mask])
            )
            normed_dot_product = np.nan_to_num(normed_dot_product)
            result.append(normed_dot_product)
        else:
            result.append(dot_product)
    return lags, np.array(result)


def cross_corr_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    maxlags: int = 21,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Cross-correlation plot between multiple timeseries.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    n_segments:
        number of random segments to plot, ignored if parameter ``segments`` is set
    maxlags:
        number of timeseries shifts for cross-correlation, should be >=1 and <= len(timeseries)
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        parameter ``maxlags`` doesn't satisfy constraints
    """
    if segments is None:
        exist_segments = list(ts.segments)
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)

    segment_pairs = list(combinations(segments, r=2))
    if len(segment_pairs) == 0:
        raise ValueError("There are no pairs to plot! Try set n_segments > 1.")

    fig, ax = _prepare_axes(num_plots=len(segment_pairs), columns_num=columns_num, figsize=figsize)
    fig.suptitle("Cross-correlation", fontsize=16)

    df = ts.to_pandas()

    for i, (segment_1, segment_2) in enumerate(segment_pairs):
        target_1 = df.loc[:, pd.IndexSlice[segment_1, "target"]]
        target_2 = df.loc[:, pd.IndexSlice[segment_2, "target"]]

        if target_1.dtype == int or target_2.dtype == int:
            warnings.warn(
                "At least one target column has integer dtype, "
                "it is converted to float in order to calculate correlation."
            )
            target_1 = target_1.astype(float)
            target_2 = target_2.astype(float)

        lags, correlations = _cross_correlation(a=target_1.values, b=target_2.values, maxlags=maxlags, normed=True)
        ax[i].plot(lags, correlations, "-o", markersize=5)
        ax[i].set_title(f"{segment_1} vs {segment_2}")
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))


def acf_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    lags: int = 21,
    partial: bool = False,
    columns_num: int = 2,
    segments: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Autocorrelation and partial autocorrelation plot for multiple timeseries.

    Notes
    -----
    `Definition of autocorrelation <https://en.wikipedia.org/wiki/Autocorrelation>`_.

    `Definition of partial autocorrelation <https://en.wikipedia.org/wiki/Partial_autocorrelation_function>`_.

    * If ``partial=False`` function works with NaNs at any place of the time-series.

    * if ``partial=True`` function works only with NaNs at the edges of the time-series and fails if there are NaNs inside it.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    n_segments:
        number of random segments to plot
    lags:
        number of timeseries shifts for cross-correlation
    partial:
        plot autocorrelation or partial autocorrelation
    columns_num:
        number of columns in subplots
    segments:
        segments to plot
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        If partial=True and there is a NaN in the middle of the time series
    """
    if segments is None:
        exist_segments = sorted(ts.segments)
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)

    title = "Partial Autocorrelation" if partial else "Autocorrelation"

    fig, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    df = ts.to_pandas()

    for i, name in enumerate(segments):
        df_slice = df[name].reset_index()["target"]
        if partial:
            # for partial autocorrelation remove NaN from the beginning and end of the series
            begin = df_slice.first_valid_index()
            end = df_slice.last_valid_index()
            x = df_slice.values[begin:end]
            if np.isnan(x).any():
                raise ValueError("There is a NaN in the middle of the time series!")
            plot_pacf(x=x, ax=ax[i], lags=lags)

        if not partial:
            plot_acf(x=df_slice.values, ax=ax[i], lags=lags, missing="conservative")

        ax[i].set_title(name)

    plt.show()


def distribution_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    segments: Optional[List[str]] = None,
    shift: int = 30,
    window: int = 30,
    freq: str = "1M",
    n_rows: int = 10,
    figsize: Tuple[int, int] = (10, 5),
):
    """Distribution of z-values grouped by segments and time frequency.

    Mean is calculated by the windows:

    .. math::
        mean_{i} = \\sum_{j=i-\\text{shift}}^{i-\\text{shift}+\\text{window}} \\frac{x_{j}}{\\text{window}}

    The same is applied to standard deviation.

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
        group for z-values
    n_rows:
        maximum number of rows to plot
    figsize:
        size of the figure per subplot with one segment in inches
    """
    df_pd = ts.to_pandas(flatten=True)

    if segments is None:
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

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True, squeeze=False)
    fig.suptitle(f"Z statistic shift: {shift} window: {window}", fontsize=16)
    ax = ax.ravel()
    i = 0
    for period, df_slice in grouped_data:
        if period not in groups:
            continue
        sns.boxplot(data=df_slice.sort_values(by="segment"), y="z", x="segment", ax=ax[i], fliersize=False)
        ax[i].set_title(f"{period}")
        ax[i].grid()
        i += 1


def plot_imputation(
    ts: "TSDataset",
    imputer: "TimeSeriesImputerTransform",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Plot the result of imputation by a given imputer.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    imputer:
        transform to make imputation of NaNs
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    start:
        start timestamp for plot
    end:
        end timestamp for plot
    """
    start, end = _get_borders_ts(ts, start, end)

    if segments is None:
        segments = sorted(ts.segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    ts_after = deepcopy(ts)
    imputer.fit_transform(ts_after)
    feature_name = imputer.in_column

    for i, segment in enumerate(segments):
        # we want to capture nans at the beginning, so don't use `ts[:, segment, :]`
        segment_before_df = ts.to_pandas().loc[start:end, pd.IndexSlice[segment, feature_name]]  # type: ignore
        segment_after_df = ts_after.to_pandas().loc[start:end, pd.IndexSlice[segment, feature_name]]  # type: ignore

        # plot result after imputation
        ax[i].plot(segment_after_df.index, segment_after_df)

        # highlight imputed points
        imputed_index = ~segment_after_df.isna() & segment_before_df.isna()
        ax[i].scatter(
            segment_after_df.loc[imputed_index].index,
            segment_after_df.loc[imputed_index],
            c="red",
            zorder=2,
        )

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
