from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.models import ProphetModel
    from etna.models import SARIMAXModel


def get_anomalies_confidence_interval(
    ts: "TSDataset",
    model: Union["ProphetModel", "SARIMAXModel"],
    interval_width: float = 0.95,
    **model_params,
) -> Dict[str, List[pd.Timestamp]]:
    """
    Get point outliers in time series using confidence intervals (estimation model-based method).
    Outliers are all points out of the confidence interval predicted with the model.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data(should contains all the necessary features)
    model:
        model for confidence interval estimation
    interval_width:
        width of the confidence interval

    Returns
    -------
    dict of outliers: Dict[str, List[pd.Timestamp]]
        dict of outliers in format {segment: [outliers_timestamps]}
    """
    outliers_per_segment = {}
    time_points = np.array(ts.index.values)
    model = model(interval_width=interval_width, **model_params)
    model.fit(ts)
    confidence_interval = model.forecast(ts, confidence_interval=True)
    for segment in ts.segments:
        segment_slice = confidence_interval[:, segment, :][segment]
        anomalies_mask = (segment_slice["target"] > segment_slice["target_upper"]) | (
            segment_slice["target"] < segment_slice["target_lower"]
        )
        outliers_per_segment[segment] = list(time_points[anomalies_mask])
    return outliers_per_segment
