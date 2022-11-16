from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.models import ProphetModel
    from etna.models import SARIMAXModel


def create_ts_by_column(ts: "TSDataset", column: str) -> "TSDataset":
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
    from etna.datasets import TSDataset

    new_df = ts[:, :, [column]]
    new_columns_tuples = [(x[0], "target") for x in new_df.columns.tolist()]
    new_df.columns = pd.MultiIndex.from_tuples(new_columns_tuples, names=new_df.columns.names)
    return TSDataset(new_df, freq=ts.freq)


def get_anomalies_prediction_interval(
    ts: "TSDataset",
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
    time_points = np.array(ts.index.values)
    model_instance = model(**model_params)
    model_instance.fit(ts_inner)
    lower_p, upper_p = [(1 - interval_width) / 2, (1 + interval_width) / 2]
    prediction_interval = model_instance.predict(
        deepcopy(ts_inner), prediction_interval=True, quantiles=[lower_p, upper_p]
    )
    for segment in ts_inner.segments:
        predicted_segment_slice = prediction_interval[:, segment, :][segment]
        actual_segment_slice = ts_inner[:, segment, :][segment]
        anomalies_mask = (actual_segment_slice["target"] > predicted_segment_slice[f"target_{upper_p:.4g}"]) | (
            actual_segment_slice["target"] < predicted_segment_slice[f"target_{lower_p:.4g}"]
        )
        outliers_per_segment[segment] = list(time_points[anomalies_mask])
    return outliers_per_segment
