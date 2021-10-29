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


def get_anomalies_confidence_interval(
    ts: "TSDataset",
    model: Union[Type["ProphetModel"], Type["SARIMAXModel"]],
    interval_width: float = 0.95,
    **model_params,
) -> Dict[str, List[pd.Timestamp]]:
    """
    Get point outliers in time series using confidence intervals (estimation model-based method).
    Outliers are all points out of the confidence interval predicted with the model.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data(should contains all the necessary features).
    model:
        Model for confidence interval estimation.
    interval_width:
       The significance level for the confidence interval. By default a 95% confidence interval is taken.

    Returns
    -------
    dict of outliers: Dict[str, List[pd.Timestamp]]
        Dict of outliers in format {segment: [outliers_timestamps]}.

    Notes
    -----
    Works only with target column.
    """
    outliers_per_segment = {}
    time_points = np.array(ts.index.values)
    model_instance = model(**model_params)
    model_instance.fit(ts)
    confidence_interval = model_instance.forecast(deepcopy(ts), confidence_interval=True, interval_width=interval_width)
    for segment in ts.segments:
        segment_slice = confidence_interval[:, segment, :][segment]
        anomalies_mask = (segment_slice["target"] > segment_slice["target_upper"]) | (
            segment_slice["target"] < segment_slice["target_lower"]
        )
        outliers_per_segment[segment] = list(time_points[anomalies_mask])
    return outliers_per_segment
