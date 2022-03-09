import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns

from etna.transforms import Transform

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.transforms.decomposition.change_points_trend import ChangePointsTrendTransform
    from etna.transforms.decomposition.detrend import LinearTrendTransform
    from etna.transforms.decomposition.detrend import TheilSenTrendTransform
    from etna.transforms.decomposition.stl import STLTransform


def prepare_axes(segments: List[str], columns_num: int, figsize: Tuple[int, int]) -> Sequence[matplotlib.axes.Axes]:
    """Prepare axes according to segments, figure size and number of columns."""
    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    _, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True)
    ax = np.array([ax]).ravel()
    return ax


def plot_forecast(
    forecast_ts: "TSDataset",
    test_ts: Optional["TSDataset"] = None,
    train_ts: Optional["TSDataset"] = None,
    segments: Optional[List[str]] = None,
    n_train_samples: Optional[int] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    prediction_intervals: bool = False,
    quantiles: Optional[Sequence[float]] = None,
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
    segments:
        segments to plot; if not given plot all the segments from forecast_df
    n_train_samples:
        length of history of train to plot
    columns_num:
        number of graphics columns
    figsize:
        size of the figure per subplot with one segment in inches
    prediction_intervals:
        if True prediction intervals will be drawn
    quantiles:
        list of quantiles to draw
    """
    if not segments:
        segments = list(set(forecast_ts.columns.get_level_values("segment")))

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    if prediction_intervals:
        cols = [
            col
            for col in forecast_ts.columns.get_level_values("feature").unique().tolist()
            if col.startswith("target_0.")
        ]
        existing_quantiles = [float(col[7:]) for col in cols]
        if quantiles is None:
            quantiles = sorted(existing_quantiles)
        else:
            non_existent = set(quantiles) - set(existing_quantiles)
            if len(non_existent):
                warnings.warn(f"Quantiles {non_existent} do not exist in forecast dataset. They will be dropped.")
            quantiles = sorted(list(set(quantiles).intersection(set(existing_quantiles))))

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
            ax[i].plot(segment_test_df.index.values, segment_test_df.target.values, color="purple", label="test")
        ax[i].plot(segment_forecast_df.index.values, segment_forecast_df.target.values, color="r", label="forecast")

        if prediction_intervals and quantiles is not None:
            alpha = np.linspace(0, 1, len(quantiles) // 2 + 2)[1:-1]
            for quantile in range(len(quantiles) // 2):
                values_low = segment_forecast_df["target_" + str(quantiles[quantile])].values
                values_high = segment_forecast_df["target_" + str(quantiles[-quantile - 1])].values
                if quantile == len(quantiles) // 2 - 1:
                    ax[i].fill_between(
                        segment_forecast_df.index.values,
                        values_low,
                        values_high,
                        facecolor="g",
                        alpha=alpha[quantile],
                        label=f"{quantiles[quantile]}-{quantiles[-quantile-1]} prediction interval",
                    )
                else:
                    values_next = segment_forecast_df["target_" + str(quantiles[quantile + 1])].values
                    ax[i].fill_between(
                        segment_forecast_df.index.values,
                        values_low,
                        values_next,
                        facecolor="g",
                        alpha=alpha[quantile],
                        label=f"{quantiles[quantile]}-{quantiles[-quantile-1]} prediction interval",
                    )
                    values_prev = segment_forecast_df["target_" + str(quantiles[-quantile - 2])].values
                    ax[i].fill_between(
                        segment_forecast_df.index.values, values_high, values_prev, facecolor="g", alpha=alpha[quantile]
                    )
            if len(quantiles) % 2 != 0:
                values = segment_forecast_df["target_" + str(quantiles[len(quantiles) // 2])].values
                ax[i].plot(
                    segment_forecast_df.index.values,
                    values,
                    "--",
                    c="orange",
                    label=f"{quantiles[len(quantiles)//2]} quantile",
                )
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
    if not segments:
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
    anomaly_dict: Dict[str, List[np.datetime64]],
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
        dictionary derived from anomaly detection function
    segments:
        segments to plot
    columns_num:
        number of subplots columns
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if not segments:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

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
        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation
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
        Method for outliers detection
    params_bounds:
        Parameters ranges of the outliers detection method. Bounds for the parameter are (min,max,step)
    figsize:
        size of the figure in inches

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
    >>> plot_anomalies_interactive(ts=ts, segment="segment_1", method=method, params_bounds=params_bounds, figsize=(20, 10)) # doctest: +SKIP
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
        dictionary with trend change points for each segment, can be derived from `etna.analysis.find_change_points`
    segments:
        segments to use
    columns_num:
        number of subplots columns
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if not segments:
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
    Parameter `transforms` is necessary because some pipelines doesn't save features in their forecasts,
    e.g. `etna.ensembles` pipelines.
    """
    if not segments:
        segments = sorted(ts.segments)

    ax = prepare_axes(segments=segments, columns_num=columns_num, figsize=figsize)

    ts_copy = deepcopy(ts)
    ts_copy.fit_transform(transforms=transforms)
    df = ts_copy.to_pandas()
    # check if feature is present in dataset
    if feature != "timestamp":
        all_features = set(df.columns.get_level_values("feature").unique())
        if feature not in all_features:
            raise ValueError("Given feature isn't present in the dataset after applying transformations")

    for i, segment in enumerate(segments):
        segment_df = df.loc[forecast_df.index, pd.IndexSlice[segment, :]][segment].reset_index()
        segment_forecast_df = forecast_df.loc[:, pd.IndexSlice[segment, :]][segment].reset_index()
        segment_df.rename(columns={"target": "y_true"}, inplace=True)
        segment_df["y_pred"] = segment_forecast_df["target"].values

        residuals = (segment_df["y_true"] - segment_df["y_pred"]).values
        feature_values = segment_df[feature].values

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
    if len(trend_transform) == 1 and isinstance(trend_transform[0], (LinearTrendTransform, TheilSenTrendTransform)):
        for seg in segments:
            linear_coeffs[seg] = ", k=" + f'{trend_transform[0].segment_transforms[seg]._linear_model.coef_[0]:g}'
    return labels, linear_coeffs


def plot_trend(
    ts: "TSDataset",
    trend_transform: Union["TrendTransformType", List["TrendTransformType"]],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot series and trend from trend transform for this series.

    If only unique transform classes are used then show their short names (without parameters). Otherwise show their full repr as label

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
    if not segments:
        segments = list(set(ts.columns.get_level_values("segment")))

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
