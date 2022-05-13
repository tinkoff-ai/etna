import math
import warnings
from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
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
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.graphics import utils
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import STL
from typing_extensions import Literal

from etna.analysis.utils import prepare_axes

if TYPE_CHECKING:
    from etna.datasets import TSDataset

plot_acf = sm.graphics.tsa.plot_acf
plot_pacf = sm.graphics.tsa.plot_pacf


def cross_corr_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    maxlags: int = 21,
    segments: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
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
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        exist_segments = list(ts.segments)
        chosen_segments = np.random.choice(exist_segments, size=min(len(exist_segments), n_segments), replace=False)
        segments = list(chosen_segments)
    segment_pairs = list(combinations(segments, r=2))
    if len(segment_pairs) == 0:
        raise ValueError("There are no pairs to plot! Try set n_segments > 1.")
    columns_num = min(2, len(segment_pairs))
    rows_num = math.ceil(len(segment_pairs) / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True, squeeze=False)
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


def sample_acf_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    lags: int = 21,
    segments: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Autocorrelation plot for multiple timeseries.


    Notes
    -----
    `Definition of autocorrelation <https://en.wikipedia.org/wiki/Autocorrelation>`_.

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
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    k = min(n_segments, len(segments))
    columns_num = min(2, k)
    rows_num = math.ceil(k / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True, squeeze=False)
    ax = ax.ravel()
    fig.suptitle("Autocorrelation", fontsize=16)
    for i, name in enumerate(sorted(np.random.choice(segments, size=k, replace=False))):
        df_slice = ts[:, name, :][name]
        plot_acf(x=df_slice["target"].values, ax=ax[i], lags=lags)
        ax[i].set_title(name)
    plt.show()


def sample_pacf_plot(
    ts: "TSDataset",
    n_segments: int = 10,
    lags: int = 21,
    segments: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Partial autocorrelation plot for multiple timeseries.

    Notes
    -----
    `Definition of partial autocorrelation <https://en.wikipedia.org/wiki/Partial_autocorrelation_function>`_.

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
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    k = min(n_segments, len(segments))
    columns_num = min(2, k)
    rows_num = math.ceil(k / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True, squeeze=False)
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
        i += 1


def stl_plot(
    ts: "TSDataset",
    period: int,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 10),
    plot_kwargs: Optional[Dict[str, Any]] = None,
    stl_kwargs: Optional[Dict[str, Any]] = None,
):
    """Plot STL decomposition for segments.

    Parameters
    ----------
    ts:
        dataset with timeseries data
    period:
        length of seasonality
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    plot_kwargs:
        dictionary with parameters for plotting, :py:meth:`matplotlib.axes.Axes.plot` is used
    stl_kwargs:
        dictionary with parameters for STL decomposition, :py:class:`statsmodels.tsa.seasonal.STL` is used
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if stl_kwargs is None:
        stl_kwargs = {}
    if segments is None:
        segments = sorted(ts.segments)

    in_column = "target"

    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(rows_num, columns_num, squeeze=False)

    df = ts.to_pandas()
    for i, segment in enumerate(segments):
        segment_df = df.loc[:, pd.IndexSlice[segment, :]][segment]
        segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()]
        decompose_result = STL(endog=segment_df[in_column], period=period, **stl_kwargs).fit()

        # start plotting
        subfigs.flat[i].suptitle(segment)
        axs = subfigs.flat[i].subplots(4, 1, sharex=True)

        # plot observed
        axs.flat[0].plot(segment_df.index, decompose_result.observed, **plot_kwargs)
        axs.flat[0].set_ylabel("Observed")

        # plot trend
        axs.flat[1].plot(segment_df.index, decompose_result.trend, **plot_kwargs)
        axs.flat[1].set_ylabel("Trend")

        # plot seasonal
        axs.flat[2].plot(segment_df.index, decompose_result.seasonal, **plot_kwargs)
        axs.flat[2].set_ylabel("Seasonal")

        # plot residuals
        axs.flat[3].plot(segment_df.index, decompose_result.resid, **plot_kwargs)
        axs.flat[3].set_ylabel("Residual")
        axs.flat[3].tick_params("x", rotation=45)


def qq_plot(
    residuals_ts: "TSDataset",
    qq_plot_params: Optional[Dict[str, Any]] = None,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot Q-Q plots for segments.

    Parameters
    ----------
    residuals_ts:
        dataset with the time series, expected to be the residuals of the model
    qq_plot_params:
        dictionary with parameters for qq plot, :py:func:`statsmodels.graphics.gofplots.qqplot` is used
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if qq_plot_params is None:
        qq_plot_params = {}
    if segments is None:
        segments = sorted(residuals_ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    residuals_df = residuals_ts.to_pandas()
    for i, segment in enumerate(segments):
        residuals_segment = residuals_df.loc[:, pd.IndexSlice[segment, "target"]]
        qqplot(residuals_segment, ax=ax[i], **qq_plot_params)
        ax[i].set_title(segment)


def prediction_actual_scatter_plot(
    forecast_df: pd.DataFrame,
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot scatter plot with forecasted/actual values for segments.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    df = ts.to_pandas()
    for i, segment in enumerate(segments):
        forecast_segment_df = forecast_df.loc[:, pd.IndexSlice[segment, "target"]]
        segment_df = df.loc[forecast_segment_df.index, pd.IndexSlice[segment, "target"]]

        # fit a linear model
        x = forecast_segment_df.values
        y = segment_df
        model = LinearRegression()
        model.fit(X=x[:, np.newaxis], y=y)
        r2 = r2_score(y_true=y, y_pred=model.predict(x[:, np.newaxis]))

        # prepare the limits of the plot, for the identity to be from corner to corner
        x_min = min(x.min(), y.min())
        x_max = max(x.max(), y.max())
        # add some space at the borders of the plot
        x_min -= 0.05 * (x_max - x_min)
        x_max += 0.05 * (x_max - x_min)
        xlim = (x_min, x_max)
        ylim = xlim

        # make plots
        ax[i].scatter(x, y, label=f"R2: {r2:.3f}")
        x_grid = np.linspace(*xlim, 100)
        ax[i].plot(x_grid, x_grid, label="identity", linestyle="dotted", color="grey")
        ax[i].plot(
            x_grid,
            model.predict(x_grid[:, np.newaxis]),
            label=f"best fit: {model.coef_[0]:.3f} x + {model.intercept_:.3f}",
            linestyle="dashed",
            color="black",
        )
        ax[i].set_title(segment)
        ax[i].set_xlabel("$\\widehat{y}$")
        ax[i].set_ylabel("$y$")
        ax[i].set_xlim(*xlim)
        ax[i].set_ylim(*ylim)
        ax[i].legend()


class SeasonalPlotAlignment(str, Enum):
    """Enum for types of alignment in a seasonal plot.

    Attributes
    ----------
    first:
        make first period full, allow last period to have NaNs in the ending
    last:
        make last period full, allow first period to have NaNs in the beginning
    """

    first = "first"
    last = "last"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} alignments are allowed"
        )


class SeasonalPlotAggregation(str, Enum):
    """Enum for types of aggregation in a seasonal plot."""

    mean = "mean"
    sum = "sum"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregations are allowed"
        )

    @staticmethod
    def _modified_nansum(series):
        """Sum values with ignoring of NaNs.

        * If there some nan: we skip them.

        * If all values equal to nan we return nan.
        """
        if np.all(np.isnan(series)):
            return np.NaN
        else:
            return np.nansum(series)

    def get_function(self):
        """Get aggregation function."""
        if self.value == "mean":
            return np.nanmean
        elif self.value == "sum":
            return self._modified_nansum


class SeasonalPlotCycle(str, Enum):
    """Enum for types of cycles in a seasonal plot."""

    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} cycles are allowed"
        )


def _get_seasonal_cycle_name(
    timestamp: pd.Series,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
) -> pd.Series:
    """Get unique name for each cycle in a series with timestamps."""
    cycle_functions: Dict[SeasonalPlotCycle, Callable[[pd.Series], pd.Series]] = {
        SeasonalPlotCycle.hour: lambda x: x.dt.strftime("%Y-%m-%d %H"),
        SeasonalPlotCycle.day: lambda x: x.dt.strftime("%Y-%m-%d"),
        SeasonalPlotCycle.week: lambda x: x.dt.strftime("%Y-%W"),
        SeasonalPlotCycle.month: lambda x: x.dt.strftime("%Y-%b"),
        SeasonalPlotCycle.quarter: lambda x: x.apply(lambda x: f"{x.year}-{x.quarter}"),
        SeasonalPlotCycle.year: lambda x: x.dt.strftime("%Y"),
    }

    if isinstance(cycle, int):
        row_numbers = pd.Series(np.arange(len(timestamp)))
        return (row_numbers // cycle + 1).astype(str)
    else:
        return cycle_functions[SeasonalPlotCycle(cycle)](timestamp)


def _get_seasonal_in_cycle_num(
    timestamp: pd.Series,
    cycle_name: pd.Series,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
    freq: str,
) -> pd.Series:
    """Get number for each point within cycle in a series of timestamps."""
    cycle_functions: Dict[Tuple[SeasonalPlotCycle, str], Callable[[pd.Series], pd.Series]] = {
        (SeasonalPlotCycle.hour, "T"): lambda x: x.dt.minute,
        (SeasonalPlotCycle.day, "H"): lambda x: x.dt.hour,
        (SeasonalPlotCycle.week, "D"): lambda x: x.dt.weekday,
        (SeasonalPlotCycle.month, "D"): lambda x: x.dt.day,
        (SeasonalPlotCycle.quarter, "D"): lambda x: (x - pd.PeriodIndex(x, freq="Q").start_time).dt.days,
        (SeasonalPlotCycle.year, "D"): lambda x: x.dt.dayofyear,
        (SeasonalPlotCycle.year, "Q"): lambda x: x.dt.quarter,
        (SeasonalPlotCycle.year, "QS"): lambda x: x.dt.quarter,
        (SeasonalPlotCycle.year, "M"): lambda x: x.dt.month,
        (SeasonalPlotCycle.year, "MS"): lambda x: x.dt.month,
    }

    if isinstance(cycle, int):
        pass
    else:
        key = (SeasonalPlotCycle(cycle), freq)
        if key in cycle_functions:
            return cycle_functions[key](timestamp)

    # in all other cases we can use numbers within each group
    cycle_df = pd.DataFrame({"timestamp": timestamp.tolist(), "cycle_name": cycle_name.tolist()})
    return cycle_df.sort_values("timestamp").groupby("cycle_name").cumcount()


def _get_seasonal_in_cycle_name(
    timestamp: pd.Series,
    in_cycle_num: pd.Series,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
    freq: str,
) -> pd.Series:
    """Get unique name for each point within the cycle in a series of timestamps."""
    if isinstance(cycle, int):
        pass
    elif SeasonalPlotCycle(cycle) == SeasonalPlotCycle.week:
        if freq == "D":
            return timestamp.dt.strftime("%a")
    elif SeasonalPlotCycle(cycle) == SeasonalPlotCycle.year:
        if freq == "M" or freq == "MS":
            return timestamp.dt.strftime("%b")

    # in all other cases we can use numbers from cycle_num
    return in_cycle_num.astype(str)


def _seasonal_split(
    timestamp: pd.Series,
    freq: str,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
) -> pd.DataFrame:
    """Create a seasonal split into cycles of a given timestamp.

    Parameters
    ----------
    timestamp:
        series with timestamps
    freq:
        frequency of dataframe
    cycle:
        period of seasonality to capture (see :py:class:`~etna.analysis.eda_utils.SeasonalPlotCycle`)

    Returns
    -------
    result: pd.DataFrame
        dataframe with timestamps and corresponding cycle names and in cycle names
    """
    cycles_df = pd.DataFrame({"timestamp": timestamp.tolist()})
    cycles_df["cycle_name"] = _get_seasonal_cycle_name(timestamp=cycles_df["timestamp"], cycle=cycle)
    cycles_df["in_cycle_num"] = _get_seasonal_in_cycle_num(
        timestamp=cycles_df["timestamp"], cycle_name=cycles_df["cycle_name"], cycle=cycle, freq=freq
    )
    cycles_df["in_cycle_name"] = _get_seasonal_in_cycle_name(
        timestamp=cycles_df["timestamp"], in_cycle_num=cycles_df["in_cycle_num"], cycle=cycle, freq=freq
    )
    return cycles_df


def _resample(df: pd.DataFrame, freq: str, aggregation: Union[Literal["sum"], Literal["mean"]]) -> pd.DataFrame:
    from etna.datasets import TSDataset

    agg_enum = SeasonalPlotAggregation(aggregation)
    df_flat = TSDataset.to_flatten(df)
    df_flat = (
        df_flat.set_index("timestamp")
        .groupby(["segment", pd.Grouper(freq=freq)])
        .agg(agg_enum.get_function())
        .reset_index()
    )
    df = TSDataset.to_dataset(df_flat)
    return df


def _prepare_seasonal_plot_df(
    ts: "TSDataset",
    freq: str,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
    alignment: Union[Literal["first"], Literal["last"]],
    aggregation: Union[Literal["sum"], Literal["mean"]],
    in_column: str,
    segments: List[str],
):
    # for simplicity we will rename our column to target
    df = ts.to_pandas().loc[:, pd.IndexSlice[segments, in_column]]
    df.rename(columns={in_column: "target"}, inplace=True)

    # remove timestamps with only nans, it is possible if in_column != "target"
    df = df[(~df.isna()).sum(axis=1) > 0]

    # make resampling if necessary
    if ts.freq != freq:
        df = _resample(df=df, freq=freq, aggregation=aggregation)

    # process alignment
    if isinstance(cycle, int):
        timestamp = df.index
        num_to_add = -len(timestamp) % cycle
        # if we want align by the first value, then we should append NaNs to timestamp
        to_add_index = None
        if SeasonalPlotAlignment(alignment) == SeasonalPlotAlignment.first:
            to_add_index = pd.date_range(start=timestamp.max(), periods=num_to_add + 1, closed="right", freq=freq)
        # if we want to align by the last value, then we should prepend NaNs to timestamp
        elif SeasonalPlotAlignment(alignment) == SeasonalPlotAlignment.last:
            to_add_index = pd.date_range(end=timestamp.min(), periods=num_to_add + 1, closed="left", freq=freq)

        df = df.append(pd.DataFrame(None, index=to_add_index)).sort_index()

    return df


def seasonal_plot(
    ts: "TSDataset",
    freq: Optional[str] = None,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ] = "year",
    alignment: Union[Literal["first"], Literal["last"]] = "last",
    aggregation: Union[Literal["sum"], Literal["mean"]] = "sum",
    in_column: str = "target",
    plot_params: Optional[Dict[str, Any]] = None,
    cmap: str = "plasma",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot each season on one canvas for each segment.

    Parameters
    ----------
    ts:
        dataset with timeseries data
    freq:
        frequency to analyze seasons:

        * if isn't set, the frequency of ``ts`` will be used;

        * if set, resampling will be made using ``aggregation`` parameter.
          If given frequency is too low, then the frequency of ``ts`` will be used.

    cycle:
        period of seasonality to capture (see :class:`~etna.analysis.eda_utils.SeasonalPlotCycle`)
    alignment:
        how to align dataframe in case of integer cycle (see :py:class:`~etna.analysis.eda_utils.SeasonalPlotAlignment`)
    aggregation:
        how to aggregate values during resampling (see :py:class:`~etna.analysis.eda_utils.SeasonalPlotAggregation`)
    in_column:
        column to use
    cmap:
        name of colormap for plotting different cycles
        (see `Choosing Colormaps in Matplotlib <https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html>`_)
    plot_params:
        dictionary with parameters for plotting, :py:meth:`matplotlib.axes.Axes.plot` is used
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if plot_params is None:
        plot_params = {}
    if freq is None:
        freq = ts.freq
    if segments is None:
        segments = sorted(ts.segments)

    df = _prepare_seasonal_plot_df(
        ts=ts,
        freq=freq,
        cycle=cycle,
        alignment=alignment,
        aggregation=aggregation,
        in_column=in_column,
        segments=segments,
    )
    seasonal_df = _seasonal_split(timestamp=df.index.to_series(), freq=freq, cycle=cycle)

    colors = plt.get_cmap(cmap)
    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)
    for i, segment in enumerate(segments):
        segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
        cycle_names = seasonal_df["cycle_name"].unique()
        for j, cycle_name in enumerate(cycle_names):
            color = colors(j / len(cycle_names))
            cycle_df = seasonal_df[seasonal_df["cycle_name"] == cycle_name]
            segment_cycle_df = segment_df.loc[cycle_df["timestamp"]]
            ax[i].plot(
                cycle_df["in_cycle_num"],
                segment_cycle_df[cycle_df["timestamp"]],
                color=color,
                label=cycle_name,
                **plot_params,
            )

        # draw ticks if they are not digits
        if not np.all(seasonal_df["in_cycle_name"].str.isnumeric()):
            ticks_dict = {key: value for key, value in zip(seasonal_df["in_cycle_num"], seasonal_df["in_cycle_name"])}
            ticks = np.array(list(ticks_dict.keys()))
            ticks_labels = np.array(list(ticks_dict.values()))
            idx_sort = np.argsort(ticks)
            ax[i].set_xticks(ticks=ticks[idx_sort], labels=ticks_labels[idx_sort])
        ax[i].set_xlabel(freq)
        ax[i].set_title(segment)
        ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6)
