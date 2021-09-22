import math
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from etna.datasets.tsdataset import TSDataset


def plot_forecast(
    forecast_ts: TSDataset,
    test_ts: TSDataset,
    train_ts: Optional[TSDataset] = None,
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
        segments = list(set(test_ts.columns.get_level_values("segment")))
    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    _, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), constrained_layout=True)
    ax = np.array([ax]).ravel()

    if train_ts is not None:
        train_ts.df.sort_values(by="timestamp", inplace=True)
    test_ts.df.sort_values(by="timestamp", inplace=True)
    forecast_ts.df.sort_values(by="timestamp", inplace=True)

    for i, segment in enumerate(segments):
        if train_ts is not None:
            segment_train_df = train_ts[:, segment, :][segment]
        else:
            segment_train_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        segment_test_df = test_ts[:, segment, :][segment]

        if n_train_samples is None:
            plot_df = segment_train_df
        elif n_train_samples != 0:
            plot_df = segment_train_df[-n_train_samples:]
        else:
            plot_df = pd.DataFrame(columns=["timestamp", "target", "segment"])

        segment_forecast_df = forecast_ts[:, segment, :][segment]

        if (train_ts is not None) and (n_train_samples != 0):
            ax[i].plot(plot_df.index.values, plot_df.target.values, label="train")
        ax[i].plot(segment_test_df.index.values, segment_test_df.target.values, label="test")
        ax[i].plot(segment_forecast_df.index.values, segment_forecast_df.target.values, label="forecast")
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend()


def plot_backtest(
    forecast_df: pd.DataFrame,
    ts: TSDataset,
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
        dataframe of timeseries that was used for TimeSeriesCrossValidation
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


def plot_anomalies(
    ts: TSDataset,
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

        anomaly = sorted(anomaly)
        ax[i].scatter(anomaly, segment_df[segment_df.index.isin(anomaly)]["target"].values, c="r")

        ax[i].tick_params("x", rotation=45)


def get_correlation_matrix(ts: TSDataset, segments: Optional[List[str]] = None, method: str = "pearson") -> np.array:
    """Compute pairwise correlation of timeseries for selected segments.

    Parameters
    -----------
    ts :
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
    ts: TSDataset, segments: Optional[List[str]] = None, method: str = "pearson", **heatmap_kwargs
):
    """Plot pairwise correlation heatmap for selected segments.

    Parameters
    -----------
    ts :
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

    correlation_matrix = get_correlation_matrix(ts, segments, method)
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".1g", square=True, **heatmap_kwargs)
    labels = list(ts[:, segments, :].columns.values)
    ax.set_xticklabels(labels, rotation=45, horizontalalignment="right")
    ax.set_yticklabels(labels, rotation=0, horizontalalignment="right")
    ax.set_title("Correlation Heatmap")
