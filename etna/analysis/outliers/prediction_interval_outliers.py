from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Type
from typing import Union

import pandas as pd

from etna.datasets import TSDataset

if TYPE_CHECKING:
    from etna.models import ProphetModel
    from etna.models import SARIMAXModel


def create_ts_by_column(ts: TSDataset, column: str) -> TSDataset:
    """Create TSDataset based on original ts with selecting only column in each segment and setting it to target.

    Parameters
    ----------
    ts:
        dataset with timeseries data
    column:
        column to select in each.

    Returns
    -------
    result: TSDataset
        dataset with selected column.
    """
    new_df = ts[:, :, [column]]
    new_columns_tuples = [(x[0], "target") for x in new_df.columns.tolist()]
    new_df.columns = pd.MultiIndex.from_tuples(new_columns_tuples, names=new_df.columns.names)
    return TSDataset(new_df, freq=ts.freq)


def _select_segments_subset(ts: TSDataset, segments: List[str]) -> TSDataset:
    """Create TSDataset with certain segments.

    Parameters
    ----------
    ts:
        dataset with timeseries data
    segments:
        list with segments names

    Returns
    -------
    result: TSDataset
        dataset with selected column.
    """
    df = ts.raw_df.loc[:, pd.IndexSlice[segments, :]].copy()
    df = df.dropna()
    df_exog = ts.df_exog
    if df_exog is not None:
        df_exog = df_exog.loc[df.index, pd.IndexSlice[segments, :]].copy()
    known_future = ts.known_future
    freq = ts.freq
    subset_ts = TSDataset(df=df, df_exog=df_exog, known_future=known_future, freq=freq)
    return subset_ts


def get_anomalies_prediction_interval(
    ts: TSDataset,
    model: Union[Type["ProphetModel"], Type["SARIMAXModel"]],
    interval_width: float = 0.95,
    in_column: str = "target",
    **model_params,
) -> Dict[str, List[pd.Timestamp]]:
    """
    Get point outliers in time series using prediction intervals (estimation model-based method).

    Outliers are all points out of the prediction interval predicted with the model.

    Parameters
    ----------
    ts:
        dataset with timeseries data(should contains all the necessary features).
    model:
        model for prediction interval estimation.
    interval_width:
        the significance level for the prediction interval. By default a 95% prediction interval is taken.
    in_column:
        column to analyze

        * If it is set to "target", then all data will be used for prediction.

        * Otherwise, only column data will be used.

    Returns
    -------
    :
        dict of outliers in format {segment: [outliers_timestamps]}.

    Notes
    -----
    For not "target" column only column data will be used for learning.
    """
    if in_column == "target":
        ts_inner = ts
    else:
        ts_inner = create_ts_by_column(ts, in_column)
    outliers_per_segment = {}
    model_instance = model(**model_params)
    model_instance.fit(ts_inner)
    lower_p, upper_p = [(1 - interval_width) / 2, (1 + interval_width) / 2]
    for segment in ts_inner.segments:
        ts_segment = _select_segments_subset(ts=ts_inner, segments=[segment])
        prediction_interval = model_instance.predict(
            deepcopy(ts_segment), prediction_interval=True, quantiles=[lower_p, upper_p]
        )
        actual_segment_slice = ts_segment[:, segment, :][segment]
        predicted_segment_slice = prediction_interval[actual_segment_slice.index, segment, :][segment]
        anomalies_mask = (actual_segment_slice["target"] > predicted_segment_slice[f"target_{upper_p:.4g}"]) | (
            actual_segment_slice["target"] < predicted_segment_slice[f"target_{lower_p:.4g}"]
        )
        outliers_per_segment[segment] = list(predicted_segment_slice[anomalies_mask].index.values)
    return outliers_per_segment
