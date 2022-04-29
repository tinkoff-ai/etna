import itertools
import math
import warnings
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

import holidays as holidays_lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal import periodogram
from typing_extensions import Literal

from etna.analysis import RelevanceTable
from etna.analysis.feature_selection import AGGREGATION_FN
from etna.analysis.feature_selection import AggregationMode
from etna.analysis.utils import prepare_axes
from etna.transforms import Transform

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.transforms import TimeSeriesImputerTransform
    from etna.transforms.decomposition.change_points_trend import ChangePointsTrendTransform
    from etna.transforms.decomposition.detrend import LinearTrendTransform
    from etna.transforms.decomposition.detrend import TheilSenTrendTransform
    from etna.transforms.decomposition.stl import STLTransform


def _get_existing_quantiles(ts: "TSDataset") -> Set[float]:
    """Get quantiles that are present inside the TSDataset."""
    cols = [col for col in ts.columns.get_level_values("feature").unique().tolist() if col.startswith("target_0.")]
    existing_quantiles = {float(col[len("target_") :]) for col in cols}
    return existing_quantiles


def _select_quantiles(forecast_results: Dict[str, "TSDataset"], quantiles: Optional[List[float]]) -> List[float]:
    """Select quantiles from the forecast results.

    Selected quantiles exist in each forecast.
    """
    intersection_quantiles_set = set.intersection(
        *[_get_existing_quantiles(forecast) for forecast in forecast_results.values()]
    )
    intersection_quantiles = sorted(list(intersection_quantiles_set))

    if quantiles is None:
        selected_quantiles = intersection_quantiles
    else:
        selected_quantiles = sorted(list(set(quantiles) & intersection_quantiles_set))
        non_existent = set(quantiles) - intersection_quantiles_set
        if non_existent:
            warnings.warn(f"Quantiles {non_existent} do not exist in each forecast dataset. They will be dropped.")

    return selected_quantiles


def _prepare_forecast_results(
    forecast_ts: Union["TSDataset", List["TSDataset"], Dict[str, "TSDataset"]]
) -> Dict[str, "TSDataset"]:
    """Prepare dictionary with forecasts results."""
    from etna.datasets import TSDataset

    if isinstance(forecast_ts, TSDataset):
        return {"1": forecast_ts}
    elif isinstance(forecast_ts, list) and len(forecast_ts) > 0:
        return {str(i + 1): forecast for i, forecast in enumerate(forecast_ts)}
    elif isinstance(forecast_ts, dict) and len(forecast_ts) > 0:
        return forecast_ts
    else:
        raise ValueError("Unknown type of `forecast_ts`")


def plot_forecast(
    forecast_ts: Union["TSDataset", List["TSDataset"], Dict[str, "TSDataset"]],
    test_ts: Optional["TSDataset"] = None,
    train_ts: Optional["TSDataset"] = None,
    segments: Optional[List[str]] = None,
    n_train_samples: Optional[int] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    prediction_intervals: bool = False,
    quantiles: Optional[List[float]] = None,
):
    """
    Plot of prediction for forecast pipeline.

    Parameters
    ----------
    forecast_ts:
        there are several options:

        #. Forecasted TSDataset with timeseries data, single-forecast mode

        #. List of forecasted TSDatasets, multi-forecast mode

        #. Dictionary with forecasted TSDatasets, multi-forecast mode

    test_ts:
        TSDataset with timeseries data
    train_ts:
        TSDataset with timeseries data
    segments:
        segments to plot; if not given plot all the segments from ``forecast_df``
    n_train_samples:
        length of history of train to plot
    columns_num:
        number of graphics columns
    figsize:
        size of the figure per subplot with one segment in inches
    prediction_intervals:
        if True prediction intervals will be drawn
    quantiles:
        List of quantiles to draw, if isn't set then quantiles from a given dataset will be used.
        In multi-forecast mode, only quantiles present in each forecast will be used.

    Raises
    ------
    ValueError:
        if the format of ``forecast_ts`` is unknown
    """
    forecast_results = _prepare_forecast_results(forecast_ts)
    num_forecasts = len(forecast_results.keys())

    if segments is None:
        unique_segments = set()
        for forecast in forecast_results.values():
            unique_segments.update(forecast.segments)
        segments = list(unique_segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    if prediction_intervals:
        quantiles = _select_quantiles(forecast_results, quantiles)

    if train_ts is not None:
        train_ts.df.sort_values(by="timestamp", inplace=True)
    if test_ts is not None:
        test_ts.df.sort_values(by="timestamp", inplace=True)

    for i, segment in enumerate(segments):
        if train_ts is not None:
            segment_train_df = train_ts[:, segment, :][segment]
        else:
            segment_train_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if test_ts is not None:
            segment_test_df = test_ts[:, segment, :][segment]
        else:
            segment_test_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if n_train_samples is None:
            plot_df = segment_train_df
        elif n_train_samples != 0:
            plot_df = segment_train_df[-n_train_samples:]
        else:
            plot_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        if (train_ts is not None) and (n_train_samples != 0):
            ax[i].plot(plot_df.index.values, plot_df.target.values, label="train")
        if test_ts is not None:
            ax[i].plot(segment_test_df.index.values, segment_test_df.target.values, color="purple", label="test")

        # plot forecast plot for each of given forecasts
        quantile_prefix = "target_"
        for j, (forecast_name, forecast) in enumerate(forecast_results.items()):
            legend_prefix = f"{forecast_name}: " if num_forecasts > 1 else ""

            segment_forecast_df = forecast[:, segment, :][segment].sort_values(by="timestamp")
            line = ax[i].plot(
                segment_forecast_df.index.values,
                segment_forecast_df.target.values,
                linewidth=1,
                label=f"{legend_prefix}forecast",
            )
            forecast_color = line[0].get_color()

            # draw prediction intervals from outer layers to inner ones
            if prediction_intervals and quantiles is not None:
                alpha = np.linspace(0, 1 / 2, len(quantiles) // 2 + 2)[1:-1]
                for quantile_idx in range(len(quantiles) // 2):
                    # define upper and lower border for this iteration
                    low_quantile = quantiles[quantile_idx]
                    high_quantile = quantiles[-quantile_idx - 1]
                    values_low = segment_forecast_df[f"{quantile_prefix}{low_quantile}"].values
                    values_high = segment_forecast_df[f"{quantile_prefix}{high_quantile}"].values
                    # if (low_quantile, high_quantile) is the smallest interval
                    if quantile_idx == len(quantiles) // 2 - 1:
                        ax[i].fill_between(
                            segment_forecast_df.index.values,
                            values_low,
                            values_high,
                            facecolor=forecast_color,
                            alpha=alpha[quantile_idx],
                            label=f"{legend_prefix}{low_quantile}-{high_quantile}",
                        )
                    # if there is some interval inside (low_quantile, high_quantile) we should plot around it
                    else:
                        low_next_quantile = quantiles[quantile_idx + 1]
                        high_prev_quantile = quantiles[-quantile_idx - 2]
                        values_next = segment_forecast_df[f"{quantile_prefix}{low_next_quantile}"].values
                        ax[i].fill_between(
                            segment_forecast_df.index.values,
                            values_low,
                            values_next,
                            facecolor=forecast_color,
                            alpha=alpha[quantile_idx],
                            label=f"{legend_prefix}{low_quantile}-{high_quantile}",
                        )
                        values_prev = segment_forecast_df[f"{quantile_prefix}{high_prev_quantile}"].values
                        ax[i].fill_between(
                            segment_forecast_df.index.values,
                            values_high,
                            values_prev,
                            facecolor=forecast_color,
                            alpha=alpha[quantile_idx],
                        )
                # when we can't find pair quantile, we plot it separately
                if len(quantiles) % 2 != 0:
                    remaining_quantile = quantiles[len(quantiles) // 2]
                    values = segment_forecast_df[f"{quantile_prefix}{remaining_quantile}"].values
                    ax[i].plot(
                        segment_forecast_df.index.values,
                        values,
                        "--",
                        color=forecast_color,
                        label=f"{legend_prefix}{remaining_quantile}",
                    )
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend(loc="upper left")


def plot_backtest(
    forecast_df: pd.DataFrame,
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    columns_num: int = 2,
    history_len: int = 0,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot targets and forecast for backtest pipeline.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    folds:
        folds to plot
    columns_num:
        number of subplots columns
    history_len:
        length of pre-backtest history to plot
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)
    df = ts.df

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    if not folds:
        folds = sorted(set(forecast_df[segments[0]]["fold_number"]))

    forecast_start = forecast_df.index.min()
    history_df = df[df.index < forecast_start]
    backtest_df = df[df.index >= forecast_start]
    for i, segment in enumerate(segments):
        segment_backtest_df = backtest_df[segment]
        segment_history_df = history_df[segment]

        if history_len:
            plot_df = segment_history_df.tail(history_len)
        else:
            plot_df = segment_backtest_df

        ax[i].plot(plot_df.index, plot_df.target, label="history")
        ax[i].plot(segment_backtest_df.index, segment_backtest_df.target, label="test")

        segment_forecast_df = forecast_df[segment]
        for fold_number in folds:
            forecast_df_slice_fold = segment_forecast_df[segment_forecast_df.fold_number == fold_number]
            ax[i].axvspan(
                forecast_df_slice_fold.index.min(),
                forecast_df_slice_fold.index.max(),
                alpha=0.15 * (int(forecast_df_slice_fold.fold_number.max() + 1) % 2),
                color="skyblue",
            )

        ax[i].plot(segment_forecast_df.index, segment_forecast_df.target, label="forecast")

        ax[i].set_title(segment)
        ax[i].legend()
        ax[i].tick_params("x", rotation=45)


def plot_backtest_interactive(
    forecast_df: pd.DataFrame,
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    history_len: int = 0,
    figsize: Tuple[int, int] = (900, 600),
) -> go.Figure:
    """Plot targets and forecast for backtest pipeline using plotly.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataframe of timeseries that was used for backtest
    segments:
        segments to plot
    folds:
        folds to plot
    history_len:
        length of pre-backtest history to plot
    figsize:
        size of the figure in pixels

    Returns
    -------
    go.Figure:
        result of plotting
    """
    if segments is None:
        segments = sorted(ts.segments)
    df = ts.df

    if not folds:
        folds = sorted(set(forecast_df[segments[0]]["fold_number"]))

    fig = go.Figure()
    colors = plotly.colors.qualitative.Dark24

    forecast_start = forecast_df.index.min()
    history_df = df[df.index < forecast_start]
    backtest_df = df[df.index >= forecast_start]

    for i, segment in enumerate(segments):
        segment_backtest_df = backtest_df[segment]
        segment_history_df = history_df[segment]

        if history_len:
            plot_df = segment_history_df.tail(history_len)
        else:
            plot_df = segment_backtest_df

        # history
        fig.add_trace(
            go.Scattergl(
                x=plot_df.index,
                y=plot_df.target,
                legendgroup=f"{segment}",
                name=f"{segment}",
                marker_color=colors[i % len(colors)],
                showlegend=True,
                line=dict(width=2, dash="solid"),
            )
        )

        # test
        fig.add_trace(
            go.Scattergl(
                x=segment_backtest_df.index,
                y=segment_backtest_df.target,
                legendgroup=f"{segment}",
                name=f"Test: {segment}",
                marker_color=colors[i % len(colors)],
                showlegend=False,
                line=dict(width=2, dash="dot"),
            )
        )

        # folds
        segment_forecast_df = forecast_df[segment]
        if i == 0:
            for fold_number in folds:
                forecast_df_slice_fold = segment_forecast_df[segment_forecast_df.fold_number == fold_number]
                opacity = 0.15 * (int(forecast_df_slice_fold.fold_number.max() + 1) % 2)
                fig.add_vrect(
                    x0=forecast_df_slice_fold.index.min(),
                    x1=forecast_df_slice_fold.index.max(),
                    line_width=0,
                    fillcolor="blue",
                    opacity=opacity,
                )

        # forecast
        fig.add_trace(
            go.Scattergl(
                x=segment_forecast_df.index,
                y=segment_forecast_df.target,
                legendgroup=f"{segment}",
                name=f"Forecast: {segment}",
                marker_color=colors[i % len(colors)],
                showlegend=False,
                line=dict(width=2, dash="dash"),
            )
        )

    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        title="Backtest for all segments",
        xaxis_title="timestamp",
        yaxis_title="target",
        legend=dict(itemsizing="trace", title="Segments"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                xanchor="left",
                yanchor="top",
                showactive=True,
                x=1.0,
                y=1.1,
                buttons=[
                    dict(method="restyle", args=["visible", "all"], label="show all"),
                    dict(method="restyle", args=["visible", "legendonly"], label="hide all"),
                ],
            )
        ],
        annotations=[
            dict(text="Show segments:", showarrow=False, x=1.0, y=1.08, xref="paper", yref="paper", align="left")
        ],
    )

    return fig


def plot_anomalies(
    ts: "TSDataset",
    anomaly_dict: Dict[str, List[pd.Timestamp]],
    in_column: str = "target",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot a time series with indicated anomalies.

    Parameters
    ----------
    ts:
        TSDataset of timeseries that was used for detect anomalies
    anomaly_dict:
        dictionary derived from anomaly detection function,
        e.g. :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density`
    in_column:
        column to plot
    segments:
        segments to plot
    columns_num:
        number of subplots columns
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    for i, segment in enumerate(segments):
        segment_df = ts[:, segment, :][segment]
        anomaly = anomaly_dict[segment]

        ax[i].set_title(segment)
        ax[i].plot(segment_df.index.values, segment_df[in_column].values, c="b")

        anomaly = sorted(anomaly)  # type: ignore
        ax[i].scatter(anomaly, segment_df[segment_df.index.isin(anomaly)][in_column].values, c="r")

        ax[i].tick_params("x", rotation=45)


def get_correlation_matrix(
    ts: "TSDataset", segments: Optional[List[str]] = None, method: str = "pearson"
) -> np.ndarray:
    """Compute pairwise correlation of timeseries for selected segments.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    segments:
        Segments to use
    method:
        Method of correlation:

        * pearson: standard correlation coefficient

        * kendall: Kendall Tau correlation coefficient

        * spearman: Spearman rank correlation

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    if method not in ["pearson", "kendall", "spearman"]:
        raise ValueError(f"'{method}' is not a valid method of correlation.")
    if segments is None:
        segments = sorted(ts.segments)
    correlation_matrix = ts[:, segments, :].corr(method=method).values
    return correlation_matrix


def plot_correlation_matrix(
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    method: str = "pearson",
    figsize: Tuple[int, int] = (10, 10),
    **heatmap_kwargs,
):
    """Plot pairwise correlation heatmap for selected segments.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    segments:
        Segments to use
    method:
        Method of correlation:

        * pearson: standard correlation coefficient

        * kendall: Kendall Tau correlation coefficient

        * spearman: Spearman rank correlation

    figsize:
        size of the figure in inches
    """
    if segments is None:
        segments = sorted(ts.segments)
    if "vmin" not in heatmap_kwargs:
        heatmap_kwargs["vmin"] = -1
    if "vmax" not in heatmap_kwargs:
        heatmap_kwargs["vmax"] = 1

    correlation_matrix = get_correlation_matrix(ts, segments, method)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".1g", square=True, ax=ax, **heatmap_kwargs)
    labels = list(ts[:, segments, :].columns.values)
    ax.set_xticklabels(labels, rotation=45, horizontalalignment="right")
    ax.set_yticklabels(labels, rotation=0, horizontalalignment="right")
    ax.set_title("Correlation Heatmap")


def plot_anomalies_interactive(
    ts: "TSDataset",
    segment: str,
    method: Callable[..., Dict[str, List[pd.Timestamp]]],
    params_bounds: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]],
    in_column: str = "target",
    figsize: Tuple[int, int] = (20, 10),
):
    """Plot a time series with indicated anomalies.

    Anomalies are obtained using the specified method. The method parameters values
    can be changed using the corresponding sliders.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    segment:
        Segment to plot
    method:
        Method for outliers detection, e.g. :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density`
    params_bounds:
        Parameters ranges of the outliers detection method. Bounds for the parameter are (min,max,step)
    in_column:
        column to plot
    figsize:
        size of the figure in inches

    Notes
    -----
    Jupyter notebook might display the results incorrectly,
    in this case try to use ``!jupyter nbextension enable --py widgetsnbextension``.

    Examples
    --------
    >>> from etna.datasets import TSDataset
    >>> from etna.datasets import generate_ar_df
    >>> from etna.analysis import plot_anomalies_interactive, get_anomalies_density
    >>> classic_df = generate_ar_df(periods=1000, start_time="2021-08-01", n_segments=2)
    >>> df = TSDataset.to_dataset(classic_df)
    >>> ts = TSDataset(df, "D")
    >>> params_bounds = {"window_size": (5, 20, 1), "distance_coef": (0.1, 3, 0.25)}
    >>> method = get_anomalies_density
    >>> plot_anomalies_interactive(ts=ts, segment="segment_1", method=method, params_bounds=params_bounds, figsize=(20, 10)) # doctest: +SKIP
    """
    from ipywidgets import FloatSlider
    from ipywidgets import IntSlider
    from ipywidgets import interact

    from etna.datasets import TSDataset

    df = ts[:, segment, in_column]
    ts = TSDataset(ts[:, segment, :], ts.freq)
    x, y = df.index.values, df.values
    cache = {}

    sliders = dict()
    style = {"description_width": "initial"}
    for param, bounds in params_bounds.items():
        min_, max_, step = bounds
        if isinstance(min_, float) or isinstance(max_, float) or isinstance(step, float):
            sliders[param] = FloatSlider(min=min_, max=max_, step=step, continuous_update=False, style=style)
        else:
            sliders[param] = IntSlider(min=min_, max=max_, step=step, continuous_update=False, style=style)

    def update(**kwargs):
        key = "_".join([str(val) for val in kwargs.values()])
        if key not in cache:
            anomalies = method(ts, **kwargs)[segment]
            anomalies = sorted(anomalies)
            cache[key] = anomalies
        else:
            anomalies = cache[key]
        plt.figure(figsize=figsize)
        plt.cla()
        plt.plot(x, y)
        plt.scatter(anomalies, y[pd.to_datetime(x).isin(anomalies)], c="r")
        plt.xticks(rotation=45)
        plt.show()

    interact(update, **sliders)


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
    rows_num = math.ceil(len(unique_clusters) / columns_num)
    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, axs = plt.subplots(rows_num, columns_num, constrained_layout=True, figsize=figsize)
    for i, cluster in enumerate(unique_clusters):
        segments = [segment for segment in segment2cluster if segment2cluster[segment] == cluster]
        h, w = i // columns_num, i % columns_num
        for segment in segments:
            segment_slice = ts[:, segment, "target"]
            axs[h][w].plot(
                segment_slice.index.values,
                segment_slice.values,
                alpha=1 / math.sqrt(len(segments)),
                c="blue",
            )
        axs[h][w].set_title(f"cluster={cluster}\n{len(segments)} segments in cluster")
        if centroids_df is not None:
            centroid = centroids_df[cluster, "target"]
            axs[h][w].plot(centroid.index.values, centroid.values, c="red", label="centroid")
        axs[h][w].legend()


def plot_time_series_with_change_points(
    ts: "TSDataset",
    change_points: Dict[str, List[pd.Timestamp]],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot segments with their trend change points.

    Parameters
    ----------
    ts:
        TSDataset with timeseries
    change_points:
        dictionary with trend change points for each segment,
        can be obtained from :py:func:`~etna.analysis.change_points_trend.search.find_change_points`
    segments:
        segments to use
    columns_num:
        number of subplots columns
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    for i, segment in enumerate(segments):
        segment_df = ts[:, segment, :][segment]
        change_points_segment = change_points[segment]

        # plot each part of segment separately
        timestamp = segment_df.index.values
        target = segment_df["target"].values
        all_change_points_segment = [pd.Timestamp(timestamp[0])] + change_points_segment + [pd.Timestamp(timestamp[-1])]
        for idx in range(len(all_change_points_segment) - 1):
            start_time = all_change_points_segment[idx]
            end_time = all_change_points_segment[idx + 1]
            selected_indices = (timestamp >= start_time) & (timestamp <= end_time)
            cur_timestamp = timestamp[selected_indices]
            cur_target = target[selected_indices]
            ax[i].plot(cur_timestamp, cur_target)

        # plot each trend change point
        for change_point in change_points_segment:
            ax[i].axvline(change_point, linestyle="dashed", c="grey")

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)


def get_residuals(forecast_df: pd.DataFrame, ts: "TSDataset") -> "TSDataset":
    """Get residuals for further analysis.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataset of timeseries that has answers to forecast

    Returns
    -------
    new_ts: TSDataset
        TSDataset with residuals in forecasts

    Raises
    ------
    KeyError:
        if segments of ``forecast_df`` and ``ts`` aren't the same

    Notes
    -----
    Transforms are taken as is from ``ts``.
    """
    from etna.datasets import TSDataset

    # find the residuals
    true_df = ts[forecast_df.index, :, :]
    if set(ts.segments) != set(forecast_df.columns.get_level_values("segment").unique()):
        raise KeyError("Segments of `ts` and `forecast_df` should be the same")
    true_df.loc[:, pd.IndexSlice[ts.segments, "target"]] -= forecast_df.loc[:, pd.IndexSlice[ts.segments, "target"]]

    # make TSDataset
    new_ts = TSDataset(df=true_df, freq=ts.freq)
    new_ts.known_future = ts.known_future
    new_ts._regressors = ts.regressors
    new_ts.transforms = ts.transforms
    new_ts.df_exog = ts.df_exog
    return new_ts


def plot_residuals(
    forecast_df: pd.DataFrame,
    ts: "TSDataset",
    feature: Union[str, Literal["timestamp"]] = "timestamp",
    transforms: Sequence[Transform] = (),
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot residuals for predictions from backtest against some feature.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataframe of timeseries that was used for backtest
    feature:
        feature name to draw against residuals, if "timestamp" plot residuals against the timestamp
    transforms:
        sequence of transforms to get feature column
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if feature isn't present in the dataset after applying transformations

    Notes
    -----
    Parameter ``transforms`` is necessary because some pipelines doesn't save features in their forecasts,
    e.g. :py:mod:`etna.ensembles` pipelines.
    """
    if segments is None:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    ts_copy = deepcopy(ts)
    ts_copy.fit_transform(transforms=transforms)
    ts_residuals = get_residuals(forecast_df=forecast_df, ts=ts_copy)
    df = ts_residuals.to_pandas()
    # check if feature is present in dataset
    if feature != "timestamp":
        all_features = set(df.columns.get_level_values("feature").unique())
        if feature not in all_features:
            raise ValueError("Given feature isn't present in the dataset after applying transformations")

    for i, segment in enumerate(segments):
        segment_forecast_df = forecast_df.loc[:, pd.IndexSlice[segment, :]][segment].reset_index()
        segment_residuals_df = df.loc[:, pd.IndexSlice[segment, :]][segment].reset_index()
        residuals = segment_residuals_df["target"].values
        feature_values = segment_residuals_df[feature].values

        # highlight different backtest folds
        if feature == "timestamp":
            folds = sorted(set(segment_forecast_df["fold_number"]))
            for fold_number in folds:
                forecast_df_slice_fold = segment_forecast_df[segment_forecast_df["fold_number"] == fold_number]
                ax[i].axvspan(
                    forecast_df_slice_fold["timestamp"].min(),
                    forecast_df_slice_fold["timestamp"].max(),
                    alpha=0.15 * (int(forecast_df_slice_fold["fold_number"].max() + 1) % 2),
                    color="skyblue",
                )

        ax[i].scatter(feature_values, residuals, c="b")

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].set_xlabel(feature)


TrendTransformType = Union[
    "ChangePointsTrendTransform", "LinearTrendTransform", "TheilSenTrendTransform", "STLTransform"
]


def _get_labels_names(trend_transform, segments):
    """If only unique transform classes are used then show their short names (without parameters). Otherwise show their full repr as label."""
    from etna.transforms.decomposition.detrend import LinearTrendTransform
    from etna.transforms.decomposition.detrend import TheilSenTrendTransform

    labels = [transform.__repr__() for transform in trend_transform]
    labels_short = [i[: i.find("(")] for i in labels]
    if len(np.unique(labels_short)) == len(labels_short):
        labels = labels_short
    linear_coeffs = dict(zip(segments, ["" for i in range(len(segments))]))
    if (
        len(trend_transform) == 1
        and isinstance(trend_transform[0], (LinearTrendTransform, TheilSenTrendTransform))
        and trend_transform[0].poly_degree == 1
    ):
        for seg in segments:
            linear_coeffs[seg] = (
                ", k=" + f"{trend_transform[0].segment_transforms[seg]._pipeline.steps[1][1].coef_[0]:g}"
            )
    return labels, linear_coeffs


def plot_trend(
    ts: "TSDataset",
    trend_transform: Union["TrendTransformType", List["TrendTransformType"]],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot series and trend from trend transform for this series.

    If only unique transform classes are used then show their short names (without parameters).
    Otherwise show their full repr as label

    Parameters
    ----------
    ts:
        dataframe of timeseries that was used for trend plot
    trend_transform:
        trend transform or list of trend transforms to apply
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = ts.segments

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)
    df = ts.df

    if not isinstance(trend_transform, list):
        trend_transform = [trend_transform]

    df_detrend = [transform.fit_transform(df.copy()) for transform in trend_transform]
    labels, linear_coeffs = _get_labels_names(trend_transform, segments)

    for i, segment in enumerate(segments):
        ax[i].plot(df[segment]["target"], label="Initial series")
        for label, df_now in zip(labels, df_detrend):
            ax[i].plot(df[segment, "target"] - df_now[segment, "target"], label=label + linear_coeffs[segment])
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend()


def plot_feature_relevance(
    ts: "TSDataset",
    relevance_table: RelevanceTable,
    normalized: bool = False,
    relevance_aggregation_mode: Union[str, Literal["per-segment"]] = AggregationMode.mean,
    relevance_params: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot relevance of the features.

    The most important features are at the top, the least important are at the bottom.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    relevance_table:
        method to evaluate the feature relevance
    normalized:
        whether obtained relevances should be normalized to sum up to 1
    relevance_aggregation_mode:
        aggregation strategy for obtained feature relevance table;
        all the strategies can be examined
        at :py:class:`~etna.analysis.feature_selection.mrmr_selection.AggregationMode`
    relevance_params:
        additional keyword arguments for the ``__call__`` method of
        :py:class:`~etna.analysis.feature_relevance.relevance.RelevanceTable`
    top_k:
        number of best features to plot, if None plot all the features
    segments:
        segments to use
    columns_num:
        if ``relevance_aggregation_mode="per-segment"`` number of columns in subplots, otherwise the value is ignored
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if relevance_params is None:
        relevance_params = {}
    if segments is None:
        segments = sorted(ts.segments)

    is_ascending = not relevance_table.greater_is_better
    features = list(set(ts.columns.get_level_values("feature")) - {"target"})
    relevance_df = relevance_table(df=ts[:, :, "target"], df_exog=ts[:, :, features], **relevance_params).loc[segments]

    if relevance_aggregation_mode == "per-segment":
        ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)
        for i, segment in enumerate(segments):
            relevance = relevance_df.loc[segment].sort_values(ascending=is_ascending)
            # warning about NaNs
            if relevance.isna().any():
                na_relevance_features = relevance[relevance.isna()].index.tolist()
                warnings.warn(
                    f"Relevances on segment: {segment} of features: {na_relevance_features} can't be calculated."
                )
            relevance = relevance.dropna()[:top_k]
            if normalized:
                relevance = relevance / relevance.sum()
            sns.barplot(x=relevance.values, y=relevance.index, orient="h", ax=ax[i])
            ax[i].set_title(f"Feature relevance: {segment}")

    else:
        relevance_aggregation_fn = AGGREGATION_FN[AggregationMode(relevance_aggregation_mode)]
        relevance = relevance_df.apply(lambda x: relevance_aggregation_fn(x[~x.isna()]))  # type: ignore
        relevance = relevance.sort_values(ascending=is_ascending)
        # warning about NaNs
        if relevance.isna().any():
            na_relevance_features = relevance[relevance.isna()].index.tolist()
            warnings.warn(f"Relevances of features: {na_relevance_features} can't be calculated.")
        # if top_k == None, all the values are selected
        relevance = relevance.dropna()[:top_k]
        if normalized:
            relevance = relevance / relevance.sum()
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        sns.barplot(x=relevance.values, y=relevance.index, orient="h", ax=ax)
        ax.set_title("Feature relevance")  # type: ignore


def plot_imputation(
    ts: "TSDataset",
    imputer: "TimeSeriesImputerTransform",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
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
    """
    if segments is None:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    ts_after = deepcopy(ts)
    ts_after.fit_transform(transforms=[imputer])
    feature_name = imputer.in_column

    for i, segment in enumerate(segments):
        # we want to capture nans at the beginning, so don't use `ts[:, segment, :]`
        segment_before_df = ts.to_pandas().loc[:, pd.IndexSlice[segment, feature_name]]
        segment_after_df = ts_after.to_pandas().loc[:, pd.IndexSlice[segment, feature_name]]

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


def plot_periodogram(
    ts: "TSDataset",
    period: float,
    amplitude_aggregation_mode: Union[str, Literal["per-segment"]] = AggregationMode.mean,
    periodogram_params: Optional[Dict[str, Any]] = None,
    segments: Optional[List[str]] = None,
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
        ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)
        for i, segment in enumerate(segments):
            segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
            segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()]
            if segment_df.isna().any():
                raise ValueError(f"Periodogram can't be calculated on segment with NaNs inside: {segment}")
            frequencies, spectrum = periodogram(x=segment_df, fs=period, **periodogram_params)
            ax[i].step(frequencies, spectrum)
            ax[i].set_xscale("log")
            ax[i].set_xlabel("Frequency")
            ax[i].set_ylabel("Power spectral density")
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
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.step(frequencies, spectrum)  # type: ignore
        ax.set_xscale("log")  # type: ignore
        ax.set_xlabel("Frequency")  # type: ignore
        ax.set_ylabel("Power spectral density")  # type: ignore
        ax.set_title("Periodogram")  # type: ignore


def _create_holidays_df(country_holidays: Type["holidays_lib.HolidayBase"], timestamp: List[pd.Timestamp]):
    holiday_names = {country_holidays.get(timestamp_value) for timestamp_value in timestamp}
    holiday_names = holiday_names.difference({None})

    holidays_dict = {}
    for holiday_name in holiday_names:
        cur_holiday_index = pd.Series(timestamp).apply(lambda x: country_holidays.get(x, "") == holiday_name)
        holidays_dict[holiday_name] = cur_holiday_index

    holidays_df = pd.DataFrame(holidays_dict)
    holidays_df.index = timestamp
    return holidays_df


def plot_holidays(
    ts: "TSDataset",
    holidays: Union[str, pd.DataFrame],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
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

        * | if DataFrame, then dataframe with holidays is expected to have timestamp index with holiday names columns.
          | In a holiday column values 0 represent absence of holiday in that timestamp, 1 represent the presence.

    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = sorted(ts.segments)

    if isinstance(holidays, str):
        holidays_df = _create_holidays_df(
            country_holidays=holidays_lib.CountryHoliday(country=holidays), timestamp=ts.index.tolist()
        )
    elif isinstance(holidays, pd.DataFrame):
        holidays_df = holidays
    else:
        raise ValueError("Parameter holidays is expected as str or pd.DataFrame")

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    df = ts.to_pandas()

    for i, segment in enumerate(segments):
        segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
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


class PerFoldAggregation(str, Enum):
    """Enum for types of aggregation in a metric per-segment plot."""

    mean = "mean"
    sum = "median"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregations are allowed"
        )

    def get_function(self):
        """Get aggregation function."""
        if self.value == "mean":
            return np.nanmean
        elif self.value == "median":
            return np.nanmedian


def plot_metric_per_segment(
    metrics_df: pd.DataFrame,
    metric_name: str,
    ascending: bool = False,
    per_fold_aggregation_mode: str = PerFoldAggregation.mean,
    top_k: Optional[int] = None,
    barplot_params: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot barplot with per-segment metrics.

    Parameters
    ----------
    metrics_df:
        dataframe with metrics calculated on the backtest
    metric_name:
        name of the metric to visualize
    ascending:

        * If True, small values at the top;

        * If False, big values at the top.

    per_fold_aggregation_mode:
        how to aggregate metrics over the folds if they aren't already aggregated
        (see :py:class:`~etna.analysis.plotters.PerFoldAggregation`)
    top_k:
        number segments to show after ordering according to ``ascending``
    barplot_params:
        dictionary with parameters for plotting, :py:func:`seaborn.barplot` is used
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if ``metric_name`` isn't present in ``metrics_df``
    NotImplementedError:
        unknown ``per_fold_aggregation_mode`` is given
    """
    if barplot_params is None:
        barplot_params = {}

    aggregation_mode = PerFoldAggregation(per_fold_aggregation_mode)

    plt.figure(figsize=figsize)

    if metric_name not in metrics_df.columns:
        raise ValueError("Given metric_name isn't present in metrics_df")

    if "fold_number" in metrics_df.columns:
        metrics_dict = (
            metrics_df.groupby("segment").agg({metric_name: aggregation_mode.get_function()}).to_dict()[metric_name]
        )
    else:
        metrics_dict = metrics_df["segment", metric_name].to_dict()[metric_name]

    segments = np.array(list(metrics_dict.keys()))
    values = np.array(list(metrics_dict.values()))
    sort_idx = np.argsort(values)
    if not ascending:
        sort_idx = sort_idx[::-1]
    segments = segments[sort_idx][:top_k]
    values = values[sort_idx][:top_k]
    sns.barplot(x=values, y=segments, orient="h", **barplot_params)
    plt.title("Metric per-segment plot")
    plt.xlabel("Segment")
    plt.ylabel(metric_name)


class MetricPlotType(str, Enum):
    """Enum for types of plot in :py:func:`~etna.analysis.plotters.metric_per_segment_distribution_plot`.

    Attributes
    ----------
    hist:
        Histogram plot, :py:func:`seaborn.histplot` is used
    box:
        Boxplot, :py:func:`seaborn.boxplot` is used
    violin:
        Violin plot, :py:func:`seaborn.violinplot` is used
    """

    hist = "hist"
    box = "box"
    violin = "violin"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} plots are allowed"
        )

    def get_function(self):
        """Get aggregation function."""
        if self.value == "hist":
            return sns.histplot
        elif self.value == "box":
            return sns.boxplot
        elif self.value == "violin":
            return sns.violinplot


def metric_per_segment_distribution_plot(
    metrics_df: pd.DataFrame,
    metric_name: str,
    per_fold_aggregation_mode: Optional[str] = None,
    plot_type: Union[Literal["hist"], Literal["box"], Literal["violin"]] = "hist",
    seaborn_params: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot per-segment metrics distribution.

    Parameters
    ----------
    metrics_df:
        dataframe with metrics calculated on the backtest
    metric_name:
        name of the metric to visualize
    per_fold_aggregation_mode:

        * If None, separate distributions for each fold will be drawn

        * If str, determines how to aggregate metrics over the folds if they aren't already aggregated
        (see :py:class:`~etna.analysis.plotters.PerFoldAggregation`)

    plot_type:
        type of plot (see :py:class:`~etna.analysis.plotters.MetricPlotType`)
    seaborn_params:
        dictionary with parameters for plotting
    figsize:
        size of the figure per subplot with one segment in inches

    Raises
    ------
    ValueError:
        if ``metric_name`` isn't present in ``metrics_df``
    NotImplementedError:
        unknown ``per_fold_aggregation_mode`` is given
    """
    if seaborn_params is None:
        seaborn_params = {}

    metrics_df = metrics_df.reset_index(drop=True)
    plot_type_enum = MetricPlotType(plot_type)
    plot_function = plot_type_enum.get_function()

    plt.figure(figsize=figsize)

    if metric_name not in metrics_df.columns:
        raise ValueError("Given metric_name isn't present in metrics_df")

    # draw plot for each fold
    if per_fold_aggregation_mode is None and "fold_number" in metrics_df.columns:
        if plot_type_enum == MetricPlotType.hist:
            plot_function(data=metrics_df, x=metric_name, hue="fold_number", **seaborn_params)
        else:
            plot_function(data=metrics_df, x="fold_number", y=metric_name, **seaborn_params)
            plt.xlabel("Fold")

    # draw one plot of aggregated data
    else:
        if "fold_number" in metrics_df.columns:
            agg_func = PerFoldAggregation(per_fold_aggregation_mode).get_function()
            metrics_df = metrics_df.groupby("segment").agg({metric_name: agg_func})

        if plot_type_enum == MetricPlotType.hist:
            plot_function(data=metrics_df, x=metric_name, **seaborn_params)
        else:
            plot_function(data=metrics_df, y=metric_name, **seaborn_params)

    plt.title("Metric per-segment distribution plot")
