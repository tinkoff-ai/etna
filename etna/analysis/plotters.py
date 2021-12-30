import math
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def plot_forecast(
    forecast_ts: "TSDataset",
    test_ts: Optional["TSDataset"] = None,
    train_ts: Optional["TSDataset"] = None,
    segments: Optional[List[str]] = None,
    n_train_samples: Optional[int] = None,
    columns_num: int = 2,
):
    """
    Plot of prediction for forecast pipeline.

    Parameters
    ----------
    forecast_ts:
        forecasted TSDataset with timeseries data
    test_ts:
        TSDataset with timeseries data
    train_ts:
        TSDataset with timeseries data
    segments: list of str, optional
        segments to plot; if not given plot all the segments from forecast_df
    n_train_samples: int, optional
        length of history of train to plot
    columns_num: int
        number of graphics columns
    """
    if not segments:
        segments = list(set(forecast_ts.columns.get_level_values("segment")))
    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    _, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True)
    ax = np.array([ax]).ravel()

    if train_ts is not None:
        train_ts.df.sort_values(by="timestamp", inplace=True)
    if test_ts is not None:
        test_ts.df.sort_values(by="timestamp", inplace=True)
    forecast_ts.df.sort_values(by="timestamp", inplace=True)

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

        segment_forecast_df = forecast_ts[:, segment, :][segment]

        if (train_ts is not None) and (n_train_samples != 0):
            ax[i].plot(plot_df.index.values, plot_df.target.values, label="train")
        if test_ts is not None:
            ax[i].plot(segment_test_df.index.values, segment_test_df.target.values, label="test")
        ax[i].plot(segment_forecast_df.index.values, segment_forecast_df.target.values, label="forecast")
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend()


def plot_backtest(
    forecast_df: pd.DataFrame,
    ts: "TSDataset",
    segments: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    columns_num: int = 2,
    history_len: int = 0,
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
    """
    if not segments:
        segments = sorted(ts.segments)
    df = ts.df
    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    if not folds:
        folds = sorted(set(forecast_df[segments[0]]["fold_number"]))

    _, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True)
    ax = np.array([ax]).ravel()

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

    Returns
    -------
    go.Figure:
        result of plotting
    """
    if not segments:
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
        height=600,
        width=900,
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
    anomaly_dict: Dict[str, List[np.datetime64]],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
):
    """Plot a time series with indicated anomalies.

    Parameters
    ----------
    ts:
        TSDataset of timeseries that was used for detect anomalies
    anomaly_dict:
        dictionary derived from anomaly detection function
    segments: list of str, optional
        segments to plot
    columns_num: int
        number of subplots columns
    """
    if not segments:
        segments = sorted(ts.segments)

    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    _, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True)
    ax = np.array([ax]).ravel()

    for i, segment in enumerate(segments):
        segment_df = ts[:, segment, :][segment]
        anomaly = anomaly_dict[segment]

        ax[i].set_title(segment)
        ax[i].plot(segment_df.index.values, segment_df["target"].values, c="b")

        anomaly = sorted(anomaly)  # type: ignore
        ax[i].scatter(anomaly, segment_df[segment_df.index.isin(anomaly)]["target"].values, c="r")

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
        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation

    Returns
    -------
    Correlation matrix
    """
    if method not in ["pearson", "kendall", "spearman"]:
        raise ValueError(f"'{method}' is not a valid method of correlation.")
    if segments is None:
        segments = sorted(ts.segments)
    correlation_matrix = ts[:, segments, :].corr(method=method).values
    return correlation_matrix


def plot_correlation_matrix(
    ts: "TSDataset", segments: Optional[List[str]] = None, method: str = "pearson", **heatmap_kwargs
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
        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation
    """
    if segments is None:
        segments = sorted(ts.segments)
    if "vmin" not in heatmap_kwargs:
        heatmap_kwargs["vmin"] = -1
    if "vmax" not in heatmap_kwargs:
        heatmap_kwargs["vmax"] = 1

    correlation_matrix = get_correlation_matrix(ts, segments, method)
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".1g", square=True, **heatmap_kwargs)
    labels = list(ts[:, segments, :].columns.values)
    ax.set_xticklabels(labels, rotation=45, horizontalalignment="right")
    ax.set_yticklabels(labels, rotation=0, horizontalalignment="right")
    ax.set_title("Correlation Heatmap")


def plot_anomalies_interactive(
    ts: "TSDataset",
    segment: str,
    method: Callable[..., Dict[str, List[pd.Timestamp]]],
    params_bounds: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]],
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
        Method for outliers detection
    params_bounds:
        Parameters ranges of the outliers detection method. Bounds for the parameter are (min,max,step)

    Notes
    -----
    Jupyter notebook might display the results incorrectly, in this case try to use '!jupyter nbextension enable --py widgetsnbextension'

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
    >>> plot_anomalies_interactive(ts=ts, segment="segment_1", method=method, params_bounds=params_bounds) # doctest: +SKIP
    """
    from ipywidgets import FloatSlider
    from ipywidgets import IntSlider
    from ipywidgets import interact

    from etna.datasets import TSDataset

    df = ts[:, segment, "target"]
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
        plt.figure(figsize=(20, 10))
        plt.cla()
        plt.plot(x, y)
        plt.scatter(anomalies, y[pd.to_datetime(x).isin(anomalies)], c="r")
        plt.xticks(rotation=45)
        plt.show()

    interact(update, **sliders)


def plot_clusters(
    ts: "TSDataset", segment2cluster: Dict[str, int], centroids_df: Optional[pd.DataFrame] = None, columns_num: int = 2
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
    """
    unique_clusters = sorted(set(segment2cluster.values()))
    rows_num = math.ceil(len(unique_clusters) / columns_num)
    fig, axs = plt.subplots(rows_num, columns_num, constrained_layout=True, figsize=(20, 5 * rows_num))
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
